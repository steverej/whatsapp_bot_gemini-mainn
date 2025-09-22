from flask import Flask, jsonify, request, Response
import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WHAT_TOKEN = os.getenv("WHAT_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PHONE_NUMBER = os.getenv("PHONE_NUMBER")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')
def ai_response(ask):
    response = model.generate_content(
        ask,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7)
    )
    return response.text

@app.route('/', methods=["GET"])
def check_webhook():
    mode = request.args.get('hub.mode')
    verify_token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')

    if mode and verify_token and mode == 'subscribe' and verify_token == VERIFY_TOKEN:
        return Response(challenge, 200)
    else:
        return Response("Webhook verification failed.", 403)

@app.route('/', methods=["POST"])
def send_message():
    if request.method == 'POST':
        body = request.get_json()
        print(body)

        # The rest of your code to process the message...
        try:
            # Check if the message is from the desired phone number
            if body["entry"][0]["changes"][0]['value']["messages"][0]["from"] == PHONE_NUMBER:
                user_question = body["entry"][0]["changes"][0]['value']["messages"][0]["text"]["body"]
                response_text = ai_response(user_question)
                
                url = "https://graph.facebook.com/v18.0/115446774859882/messages"
                headers = {
                    "Authorization": f"Bearer {WHAT_TOKEN}",
                    "Content-Type": "application/json"
                }
                data = {
                    "messaging_product": "whatsapp",
                    "to": PHONE_NUMBER,
                    "type": "text",
                    "text": {"body": response_text}
                }
                
                # Send the message and handle the response
                requests.post(url, json=data, headers=headers)
                print("Message sent successfully")
                return Response(status=200) # <-- Successful path
            
            else:
                # If the sender is not the specified phone number, just return 200 to acknowledge receipt.
                return Response(status=200)
        
        except (KeyError, IndexError) as e:
            # Log the error for debugging
            print(f"Error processing webhook payload: {e}")
            # In case of a malformed request, still return a 200 OK to prevent re-delivery attempts.
            return Response(status=200)
    
    # This return is for cases where request.method is not 'POST', though the decorator prevents this.
    # It's good practice to have a final return for all paths.
    return Response(status=405) # Method Not Allowed




if __name__ == '__main__':
	app.run(host='0.0.0.0',port=os.environ.get("PORT", 5000))

