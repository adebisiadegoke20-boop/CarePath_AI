import os
from dotenv import load_dotenv

load_dotenv()

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")

class ReminderService:
    """Manages SMS reminder alerts for Garki Hospital Abuja OB/GYN appointments."""

    @classmethod
    def send_sms_reminder(cls, to_phone: str, appointment_date: str) -> dict:
        """Sends an appointment notification SMS via Twilio or simulates it if missing credentials."""
        
        message_body = (
            f"Reminder from Garki Hospital Abuja OB/GYN Department: "
            f"You have an upcoming appointment scheduled for {appointment_date}. "
            f"Please arrive 15 minutes early. For acute emergencies, report directly to the ER."
        )

        # Check if Twilio environment variables are configured
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
            print("\n[Mock Twilio Reminder Service]")
            print(f"To: {to_phone}")
            print(f"Body: {message_body}")
            print("Status: Simulated (Twilio credentials not configured in .env)\n")
            return {
                "status": "simulated",
                "to": to_phone,
                "appointment_date": appointment_date,
                "message": message_body,
                "note": "Twilio credentials missing. Add TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER to .env to send real SMS."
            }

        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            message = client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE_NUMBER,
                to=to_phone
            )
            
            return {
                "status": "sent",
                "sid": message.sid,
                "to": to_phone,
                "appointment_date": appointment_date
            }
        except Exception as e:
            print(f"Twilio Error: {e}")
            return {
                "status": "error",
                "message": f"Failed to send SMS via Twilio: {str(e)}"
            }


# ============================================================
# MODULE SELF-TEST
# ============================================================
if __name__ == "__main__":
    print("Testing Reminder Service...")
    print("-" * 40)
    result = ReminderService.send_sms_reminder(
        to_phone="+2349132072219",
        appointment_date="2026-10-25 at 10:00 AM"
    )
    print(result)