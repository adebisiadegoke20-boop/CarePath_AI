import os

# Limit thread pre-allocation to keep RAM usage well below Render's 512MB limit
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import uvicorn
from fastapi import FastAPI, Form, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.messaging_response import MessagingResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Import your multi-agent chatbot core logic
from chatbot import handle_user_message

# Load environment variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="CarePath - Garki Hospital Abuja OB/GYN Triage System",
    description="Multi-agent clinical triage system with Supabase RAG and Twilio SMS integration.",
    version="1.1.0"
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
    phone_number: Optional[str] = "web-user"  # Made optional so requests without phone don't fail 422
    message: str
    language: Optional[str] = "en"  # Language option (en, ha, yo, ig, pcm)


@app.get("/")
def read_root():
    return {
        "system": "CarePath",
        "hospital": "Garki Hospital Abuja",
        "department": "OB/GYN Clinical Triage & Patient Communication",
        "status": "Online & Operational"
    }


@app.head("/")
def read_root_head():
    return Response(status_code=200)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


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
        # Run message through your multi-agent pipeline (defaulting to English for SMS)
        result = handle_user_message(user_message, target_language="en")
        reply_text = result.get("reply", "Thank you for contacting Garki Hospital OB/GYN.")
    except Exception as e:
        print(f"Twilio Webhook Error: {str(e)}")
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
    Accepts message and optional language context.
    """
    try:
        user_msg = payload.message.strip() if payload.message else ""
        if not user_msg:
            raise HTTPException(status_code=400, detail="Message body cannot be empty.")

        target_lang = payload.language or "en"
        sender_id = payload.phone_number or "web-user"

        # Pass both message and the selected language tag to your backend
        result = handle_user_message(user_msg, target_language=target_lang)
        
        return {
            "status": result.get("status", "success"),
            "sender": sender_id,
            "category": result.get("category", "GENERAL"),
            "agent": result.get("agent", "CarePath Assistant"),
            "language": target_lang,
            "reply": result.get("reply", "Thank you for contacting Garki Hospital OB/GYN.")
        }
    except Exception as e:
        print(f"Chat Endpoint Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    server_port: int = int(os.environ.get("PORT", "10000"))
    uvicorn.run("api:app", host="0.0.0.0", port=server_port)