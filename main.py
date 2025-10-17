import json
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import threading
import os
from functools import lru_cache
import time
import pytz
import tempfile
from datetime import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# ======================================================
# 🔒 Configuration (Loading from Environment Variables)
# ======================================================
# Load secrets from environment variables. 
# IMPORTANT: You must set these variables in your deployment environment (e.g., Cloud Run, Heroku).
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", "YOUR_WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID", "YOUR_PHONE_NUMBER_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
FIREBASE_CREDENTIALS_JSON = os.environ.get("FIREBASE_CREDENTIALS_JSON")

# --- Firebase Initialization ---
if FIREBASE_CREDENTIALS_JSON:
    # Load credentials from the JSON string environment variable
    cred_dict = json.loads(FIREBASE_CREDENTIALS_JSON)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    firestore_db = firestore.client()
else:
    # Fallback or error handling for local development/missing env var
    print("WARNING: FIREBASE_CREDENTIALS_JSON not set. Firestore will not function.")
    firestore_db = None

# --- Gemini AI Configuration ---
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash', generation_config={
        "temperature": 0.3,
        "max_output_tokens": 200
    })
else:
    print("WARNING: GEMINI_API_KEY not set. AI functions will fail.")
    model = None

# --- Load Trained Knowledge Base (Local + Free) ---
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# Ensure the 'knowledge_db' directory exists and is accessible
knowledge_db = Chroma(persist_directory="knowledge_db", embedding_function=embeddings)

app = Flask(__name__)

# --- Cache for user info (5 minute TTL) ---
user_cache = {}
CACHE_TTL = 300  # 5 minutes
# @lru_cache(maxsize=1) is misplaced here, removed.

# ======================================================
# 🔍 HELPER: Retrieve relevant info from knowledge base
# ======================================================
def search_knowledge(query, top_k=3):
    """Fetches the top matching chunks from the local knowledge base."""
    if not knowledge_db:
        return "Knowledge base not initialized."
    try:
        results = knowledge_db.similarity_search(query, k=top_k)
        if not results:
            return "No relevant information found in the knowledge base."
        return "\n".join([r.page_content for r in results])
    except Exception as e:
        print(f"Error retrieving knowledge: {e}")
        return "Knowledge retrieval error."


# ======================================================
# 🔐 FIRESTORE HELPERS
# ======================================================
def get_user_info(phone_number):
    if not firestore_db: return None
    cache_key = phone_number
    if cache_key in user_cache:
        cached_data, timestamp = user_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return cached_data
    
    try:
        # Simplified logic for phone number normalization for demonstration
        query_number = phone_number.replace('+', '').lstrip('91') 
        users_ref = firestore_db.collection('app_users')
        query = users_ref.where('phone', '==', query_number).where('uid', '>', '').limit(1)
        results = query.stream()
        doc_list = list(results)
        
        if doc_list:
            doc = doc_list[0]
            user_data = doc.to_dict()
            user_data['id'] = doc.id
            user_cache[cache_key] = (user_data, time.time())
            return user_data
        else:
            user_cache[cache_key] = (None, time.time())
            return None
            
    except Exception as e:
        print(f"Error fetching user: {e}")
        return None


def get_user_bookings(uid):
    if not firestore_db: return []
    bookings_list = []
    try:
        # Use a Collection Group Query if 'bookings' is a subcollection in multiple user documents
        bookings_ref = firestore_db.collection_group('bookings').where('uid', '==', uid)
        results = bookings_ref.stream()
        for doc in results:
            bookings_list.append(doc.to_dict())
        return bookings_list
    except Exception as e:
        print(f"Error fetching bookings: {e}")
        return []

# --- Custom Booking Formatting Functions (Retained) ---
def format_bookings_for_whatsapp(bookings):
    """Formats upcoming bookings.""" 
    if not bookings: 
        return "You have no appointments found in our system." 
    upcoming = [b for b in bookings if b.get('bookingStatus') == 'upcoming'] 
    if not upcoming: 
        return "You have no upcoming appointments scheduled at this time." 
    IST = pytz.timezone('Asia/Kolkata') 
    timestamp_formats = ["%B %d, %Y at %I:%M:%S %p UTC+5:30", "%b %d, %Y at %I:%M:%S %p UTC+5:30"] 
    
    def get_sort_key(b): 
        val = b.get('timestamp') 
        if isinstance(val, datetime): return val 
        if isinstance(val, str): 
            for fmt in timestamp_formats: 
                try: return datetime.strptime(val, fmt) 
                except ValueError: continue 
        return datetime.min 
    
    upcoming.sort(key=get_sort_key) 
    response_text = "🗓️ *Your Upcoming Appointments*\n--------------------------------------\n" 
    for booking in upcoming: 
        clinic = booking.get('clinicName', 'Not specified') 
        specialization = booking.get('specialization', 'Not specified') 
        doctor = booking.get('doctorName', 'Not specified') 
        ts_val = booking.get('timestamp') 
        timestamp_str = "Not scheduled" 
        if isinstance(ts_val, datetime): 
            timestamp_str = ts_val.astimezone(IST).strftime("%A, %B %d, %Y at %I:%M %p") 
        elif isinstance(ts_val, str): 
            timestamp_str = ts_val 
        response_text += ( f"\n*Clinic:* {clinic}" f"\n*Specialization:* {specialization}" f"\n*Doctor:* {doctor}" f"\n*Date:* {timestamp_str}" f"\n--------------------------------------\n" ) 
    return response_text

def format_past_bookings_for_whatsapp(bookings):
    """Formats past bookings.""" 
    if not bookings: 
        return "You have no booking history found in our system." 
    past = [b for b in bookings if b.get('bookingStatus') == 'completed'] 
    if not past: 
        return "No completed bookings found in your history." 
    IST = pytz.timezone('Asia/Kolkata') 
    timestamp_formats = ["%B %d, %Y at %I:%M:%S %p UTC+5:30", "%b %d, %Y at %I:%M:%S %p UTC+5:30"] 
    
    def get_sort_key(b): 
        val = b.get('timestamp') 
        if isinstance(val, datetime): return val 
        if isinstance(val, str): 
            for fmt in timestamp_formats: 
                try: return datetime.strptime(val, fmt) 
                except ValueError: continue 
        return datetime.min 
    
    past.sort(key=get_sort_key, reverse=True) 
    response_text = "📋 *Your Booking History*\n--------------------------------------\n" 
    for booking in past: 
        clinic = booking.get('clinicName', 'Not specified') 
        doctor = booking.get('doctorName', 'Not specified') 
        status = booking.get('bookingStatus', 'N/A').capitalize() 
        booking_date = booking.get('bookingDate', 'Not specified') 
        response_text += ( f"\n*Clinic:* {clinic}" f"\n*Doctor:* {doctor}" f"\n*Date:* {booking_date}" f"\n*Status:* {status}" f"\n--------------------------------------\n" ) 
    return response_text

# --- Transcription Helper (Retained) ---
def transcribe_audio(audio_id): 
    """Downloads and transcribes audio from WhatsApp.""" 
    if not model: return None
    url = f"https://graph.facebook.com/v19.0/{audio_id}" 
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"} 
    temp_file_path = None 
    try: 
        response = requests.get(url, headers=headers, timeout=10) 
        response.raise_for_status() 
        media_url = response.json().get("url") 
        if not media_url: return None 
        audio_response = requests.get(media_url, headers=headers, timeout=10) 
        audio_response.raise_for_status() 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_audio_file: 
            temp_audio_file.write(audio_response.content) 
            temp_file_path = temp_audio_file.name 
        audio_file = genai.upload_file(path=temp_file_path, mime_type="audio/ogg") 
        response = model.generate_content(["Transcribe this audio message.", audio_file]) 
        genai.delete_file(audio_file.name) 
        return response.text.strip() 
    except Exception as e: 
        print(f"Error during transcription: {e}") 
        return None 
    finally: 
        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)


# ======================================================
# 🧠 AI RESPONSE HANDLER (Single AI Call)
# ======================================================
def get_smart_response(user_message, user_info=None):
    """Handles both intent detection and general Q&A with local knowledge."""
    if not model: return "general", "AI is not configured. Please check your API key."

    user_context = "User not registered."
    if user_info:
        user_context = f"User Name: {user_info.get('name', 'N/A')}, UID: {user_info.get('uid', 'N/A')}"

    # 🔍 Get local knowledge for context
    local_knowledge = search_knowledge(user_message)

    prompt = f"""
    You are ZappQ Support Assistant.

    IMPORTANT: Respond in this format:
    INTENT: [upcoming_bookings / past_bookings / user_name / general]
    RESPONSE: [your reply]

    Rules:
    - For 'general', use this Knowledge Base info:
      {local_knowledge}
    - Reply in the user's language if detected (e.g., Malayalam).
    - Always be polite and professional.
    - Keep it short and clear.

    User Context: {user_context}
    User Message: "{user_message}"
    """

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()

        lines = result.split('\n', 1)
        intent_line = lines[0] if lines else ""
        response_line = lines[1] if len(lines) > 1 else ""
        intent = "general"
        if "INTENT:" in intent_line:
            intent = intent_line.split("INTENT:")[1].strip().lower()
        answer = response_line.replace("RESPONSE:", "").strip()
        return intent, answer
    except Exception as e:
        print(f"AI Error: {e}")
        return "general", "I'm having trouble connecting right now."


# ======================================================
# 📲 WHATSAPP MESSAGE HANDLERS
# ======================================================
def send_whatsapp_message(to_number, message_text):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_text},
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        print(f"Message sent: {response.status_code}")
    except Exception as e:
        print(f"Error sending message: {e}")


def handle_user_query(user_message, user_info, from_number):
    """Handles user queries with optimized single AI call, using custom formatters."""
    
    if user_message.strip().lower() == "need help?":
        user_name = user_info.get("name", "there") if user_info else "there"
        welcome = f"Hello {user_name}! I'm the ZappQ Support assistant. How can I help you today?"
        send_whatsapp_message(from_number, welcome)
        return

    # **CRITICAL FIX: Unpack the tuple from get_smart_response**
    intent, answer = get_smart_response(user_message, user_info) 
    print(f"Intent: {intent}")

    reply = answer # Default reply is the AI's general answer

    if intent == "upcoming_bookings":
        if user_info and user_info.get("uid"):
            bookings = get_user_bookings(user_info["uid"])
            reply = format_bookings_for_whatsapp(bookings)
        else:
            reply = "I couldn't find your records to check for upcoming appointments."
    elif intent == "past_bookings":
        if user_info and user_info.get("uid"):
            bookings = get_user_bookings(user_info["uid"])
            reply = format_past_bookings_for_whatsapp(bookings)
        else:
            reply = "I couldn't find your records to check for past appointments."
    elif intent == "user_name":
        if user_info and user_info.get("name"):
            reply = f"Based on your phone number, your name is *{user_info.get('name')}*."
        else:
            reply = "I couldn't find a name associated with this phone number."
    
    send_whatsapp_message(from_number, reply)


def process_whatsapp_message(data):
    """Processes WhatsApp messages in background thread."""
    print("Processing message...")
    start_time = time.time()
    
    try:
        if data["entry"][0]["changes"][0]["value"].get("messages"):
            message_info = data["entry"][0]["changes"][0]["value"]["messages"][0]
            from_number = message_info["from"]
            message_type = message_info["type"]
            
            # Fetch user info (cached)
            user_info = get_user_info(from_number)
            
            final_text = None
            
            if message_type == 'text':
                final_text = message_info.get("text", {}).get("body", "")
                handle_user_query(final_text, user_info, from_number)

            elif message_type == 'audio':
                audio_id = message_info.get("audio", {}).get("id")
                
                # Send immediate confirmation that audio is being processed
                send_whatsapp_message(from_number, "🎤 Processing your voice message...") 
                
                final_text = transcribe_audio(audio_id)
                
                if final_text:
                    handle_user_query(final_text, user_info, from_number)
                else:
                    send_whatsapp_message(from_number, "Sorry, I couldn't understand that audio. Please type your message.")
            
            total_time = time.time() - start_time
            print(f"Total processing time: {total_time:.2f}s")

    except Exception as e:
        print(f"Error processing message: {e}")


# ======================================================
# 🌐 FLASK ROUTES
# ======================================================
@app.route("/webhook", methods=["GET"])
def webhook_verify():
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"): 
        if request.args.get("hub.verify_token") == VERIFY_TOKEN: 
            return request.args.get("hub.challenge") 
        return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def webhook_handler():
    data = request.get_json()
    if data.get("object") == "whatsapp_business_account":
        # Process the message in a non-blocking thread
        thread = threading.Thread(target=process_whatsapp_message, args=(data,))
        thread.daemon = True
        thread.start()
    return "OK", 200


@app.route("/health", methods=["GET"])
def health():
    # Simple check for application status
    status = "OK"
    if not firestore_db:
        status = "WARNING: Firestore not initialized"
    if not model:
        status = "WARNING: Gemini Model not initialized"
        
    return status, 200 if status == "OK" else 503

# ======================================================
# 🚀 FLASK RUN
# ======================================================
if __name__ == "__main__":
    # For local testing. Use environment variable PORT if available.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))