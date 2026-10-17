import os
import uvicorn
from fastapi import FastAPI, Form, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.messaging_response import MessagingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import your multi-agent chatbot core logic
from chatbot import handle_user_message

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="CarePath AI - Garki Hospital Abuja OB/GYN Triage System",
    description="Multi-agent clinical triage system with Supabase RAG and Twilio SMS integration.",
    version="1.0.0"
)

# Enable CORS for frontend/API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for web-based chat/triage requests (Swagger UI / Web App)
class ChatRequest(BaseModel):
    phone_number: str
    message: str


@app.get("/")
def read_root():
    """Health check endpoint for Render deployment."""
    return {
        "system": "CarePath AI",
        "hospital": "Garki Hospital Abuja",
        "department": "OB/GYN Clinical Triage & Patient Communication",
        "status": "Online & Operational"
    }


@app.post("/sms")
async def twilio_sms_webhook(From: str = Form(...), Body: str = Form(...)):
    """
    Twilio Webhook Endpoint:
    Listens for incoming SMS from patients, processes the query through
    the CarePath multi-agent chatbot engine, and returns a TwiML SMS XML response.
    """
    patient_phone = From
    user_message = Body

    try:
        # Run message through your multi-agent pipeline
        result = handle_user_message(user_message)
        reply_text = result.get("reply", "Thank you for contacting Garki Hospital OB/GYN.")
    except Exception as e:
        reply_text = (
            "Thank you for contacting Garki Hospital OB/GYN. "
            "Our team is experiencing high volume. Please call the emergency unit if urgent."
        )

    # Generate Twilio XML (TwiML) response
    twiml = MessagingResponse()
    twiml.message(reply_text)

    return Response(content=str(twiml), media_type="application/xml")


@app.post("/chat")
async def web_chat_endpoint(payload: ChatRequest):
    """
    Web API endpoint for testing chat and triage workflows via Swagger UI or frontend apps.
    """
    try:
        result = handle_user_message(payload.message)
        return {
            "status": result.get("status"),
            "sender": payload.phone_number,
            "category": result.get("category"),
            "agent": result.get("agent"),
            "reply": result.get("reply")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    server_port: int = int(os.environ.get("PORT", "10000"))
    uvicorn.run("api:app", host="0.0.0.0", port=server_port)