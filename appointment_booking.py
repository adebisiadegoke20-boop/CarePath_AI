import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Missing Supabase environment variables in .env file.")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
print("Appointment Booking Module Initialized")

class AppointmentBooking:
    """Manages scheduling and database records for Garki Hospital Abuja OB/GYN Department."""

    @staticmethod
    def book_appointment(patient_name: str, phone_number: str, preferred_date: str, notes: str = "") -> dict:
        """Inserts an appointment record into the Supabase appointments table."""
        try:
            payload = {
                "patient_name": patient_name,
                "phone_number": phone_number,
                "facility_name": "Garki Hospital Abuja",
                "department": "Obstetrics & Gynecology (OB/GYN)",
                "preferred_date": preferred_date,
                "notes": notes,
                "status": "confirmed"
            }
            
            response = supabase.table("appointments").insert(payload).execute()
            
            if response.data:
                return {
                    "status": "success",
                    "message": f"Appointment successfully booked at Garki Hospital Abuja OB/GYN for {preferred_date}.",
                    "data": response.data[0]
                }
            return {"status": "error", "message": "Failed to create appointment record in database."}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def get_appointments(phone_number: str = None) -> list:
        """Retrieves appointments, optionally filtered by patient phone number."""
        try:
            query = supabase.table("appointments").select("*")
            if phone_number:
                query = query.eq("phone_number", phone_number)
            response = query.execute()
            return response.data or []
        except Exception as e:
            print(f"Error fetching appointments: {e}")
            return []