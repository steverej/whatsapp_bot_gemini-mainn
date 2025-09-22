from flask import Flask, jsonify, request, Response
import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API keys and tokens
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WHAT_TOKEN = os.getenv("WHAT_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PHONE_NUMBER = os.getenv("PHONE_NUMBER")

logging.info("Environment variables loaded successfully")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro')
logging.info("Gemini AI model configured")

def ai_response(ask):
    logging.info(f"Generating AI response for: {ask}")
    response = model.generate_content(
        ask,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7
        )
    )
    logging.info(f"AI response generated: {response.text}")
    return response.text

@app.route('/', methods=["GET"])
def check_webhook():
    logging.info("Received GET request for webhook verification")
    mode = request.args.get('hub.mode')
    verify_token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    logging.info(f"Mode: {mode}, Verify Token: {verify_token}, Challenge: {challenge}")

    if mode and verify_token and mode == 'subscribe' and verify_token == VERIFY_TOKEN:
        logging.info("Webhook verified successfully")
        return Response(challenge, 200)
    else:
        logging.warning("Webhook verification failed")
        return Response("Webhook verification failed.", 403)

@app.route('/', methods=["POST"])
def send_message():
    logging.info("Received POST request")
    body = request.get_json()
    logging.info(f"Webhook payload: {body}")

    try:
        sender = body["entry"][0]["changes"][0]['value']["messages"][0]["from"]
        logging.info(f"Message received from: {sender}")

        if sender == PHONE_NUMBER:
            user_question = body["entry"][0]["changes"][0]['value']["messages"][0]["text"]["body"]
            logging.info(f"User question: {user_question}")

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

            resp = requests.post(url, json=data, headers=headers)
            logging.info(f"Message sent to WhatsApp. Response status: {resp.status_code}, body: {resp.text}")
            return Response(status=200)
        else:
            logging.info(f"Ignored message from unknown sender: {sender}")
            return Response(status=200)

    except (KeyError, IndexError) as e:
        logging.error(f"Error processing webhook payload: {e}")
        return Response(status=200)

    return Response(status=405)

if __name__ == '__main__':
    port = os.environ.get("PORT", 5000)
    logging.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=int(port))
