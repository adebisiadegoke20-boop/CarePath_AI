import os
import json
from dotenv import load_dotenv
from groq import Groq

from guard import InputGuard, OutputGuard, tracker
from triage_engine import TriageEngine
from carepath_rag_engine import full_rag
from appointment_booking import AppointmentBooking

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Language mappings for patient response translation
LANGUAGE_NAMES = {
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "pcm": "Nigerian Pidgin English",
    "en": "English"
}


def auto_detect_and_translate(user_question: str, english_response: str, target_language_code: str = "en") -> str:
    """
    Detects the language of the user's question (or uses the explicitly selected language code)
    and translates the clinical recommendation into that exact same language.
    """
    if target_language_code == "en" and user_question.isascii():
        return english_response

    target_hint = LANGUAGE_NAMES.get(target_language_code, "English") if target_language_code != "en" else "Auto-detect"

    prompt = f"""You are an empathetic medical translator at Garki Hospital Abuja.
Task:
1. Detect the language of this patient message: "{user_question}"
   (Supported languages: English, Hausa, Yoruba, Igbo, Nigerian Pidgin English).
   (User Language Dropdown Hint: {target_hint}).
2. Translate the following clinical recommendation into that exact same detected language.
   Keep it natural, polite, female-gender appropriate for OB/GYN patients, and culturally accessible in Nigeria.

Clinical Recommendation to translate:
"{english_response}"

Return ONLY the translated text without extra conversational filler or quotation marks.
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        translated_text = completion.choices[0].message.content.strip()
        return translated_text if translated_text else english_response
    except Exception as e:
        print("Translation Error:", str(e))
        return english_response


# ============================================================
# INDIVIDUAL AGENT HANDLERS
# ============================================================

def triage_agent(question):
    """Evaluates symptoms and classifies clinical urgency for Garki Hospital Abuja."""
    result = TriageEngine.evaluate_symptoms(question)
    if not result["success"]:
        return {"success": False, "response": result["error"]}
    
    triage_data = result["triage_result"]
    clean_recommendation = triage_data.get('recommendation', 'Please visit Garki Hospital Abuja OB/GYN Department.')
    return {"success": True, "response": clean_recommendation}


def appointment_agent(question):
    """Handles Garki Hospital OB/GYN appointment booking inquiries."""
    rag_res = full_rag(question)
    answer = rag_res.get("answer", "To schedule an OB/GYN appointment at Garki Hospital Abuja, please visit our clinic reception desk Monday through Friday (8:00 AM – 4:00 PM).")
    return {"success": True, "response": answer}


def clinical_rag_agent(question):
    """Handles general clinical questions using WHO and Nigerian guidelines."""
    rag_res = full_rag(question)
    answer = rag_res.get("answer", "For general inquiries, please consult the clinical team at Garki Hospital Abuja OB/GYN Department.")
    return {"success": True, "response": answer}


# ============================================================
# AGENT CLASS
# ============================================================

class Agent:
    def __init__(self, name, category, description):
        self.name = name
        self.category = category
        self.description = description

    def run(self, question):
        if self.name == "Triage & Emergency Agent":
            return triage_agent(question)
        elif self.name == "Appointment Booking Agent":
            return appointment_agent(question)
        elif self.name == "Clinical Knowledge Agent":
            return clinical_rag_agent(question)
        else:
            return {
                "success": False,
                "response": "Unknown agent."
            }


# ============================================================
# AGENTS REGISTRY
# ============================================================

triage_agent_instance = Agent(
    name="Triage & Emergency Agent",
    category="TRIAGE",
    description="Handles obstetric warning signs, symptom evaluation, and emergency triage routing for Garki Hospital Abuja."
)

appointment_agent_instance = Agent(
    name="Appointment Booking Agent",
    category="APPOINTMENT",
    description="Handles antenatal and OB/GYN appointment scheduling policies and requirements at Garki Hospital Abuja."
)

clinical_agent_instance = Agent(
    name="Clinical Knowledge Agent",
    category="CLINICAL_RAG",
    description="Handles general WHO ANC guidelines, nutrition, newborn care, and maternal health questions."
)

AGENTS = {
    "TRIAGE": triage_agent_instance,
    "APPOINTMENT": appointment_agent_instance,
    "CLINICAL_RAG": clinical_agent_instance
}


# ============================================================
# ROUTER / CLASSIFIER
# ============================================================

def classify_question(question):
    check = InputGuard.validate(question)
    if not check["valid"]:
        return "INVALID"

    q = question.lower()

    triage_keywords = [
        "pain", "bleeding", "blood", "emergency", "danger", "fever", 
        "convulsion", "convulsions", "fits", "headache", "vision", 
        "swelling", "leak", "leaking", "fluid", "vomiting", "urgent", "symptoms",
        "ciwo", "jini", "zazzabi", "ori-fifo", "eje", "iba", "isi-owuwa", "obara"
    ]
    if any(keyword in q for keyword in triage_keywords):
        return "TRIAGE"

    appointment_keywords = [
        "appointment", "book", "booking", "schedule", "scheduling", 
        "date", "time", "visit", "consultation", "clinic day"
    ]
    if any(keyword in q for keyword in appointment_keywords):
        return "APPOINTMENT"

    return "CLINICAL_RAG"


# ============================================================
# HANDLE USER MESSAGE
# ============================================================

def handle_user_message(question: str, target_language: str = "en"):
    print("\n" + "=" * 60)
    print("USER MESSAGE")
    print("=" * 60)
    print(f"Message: {question}")
    print(f"Target Language Code: {target_language}")

    # 1. Input Guard
    input_check = InputGuard.validate(question)
    if not input_check["valid"]:
        return {
            "status": "rejected",
            "category": "INVALID",
            "agent": None,
            "reply": f"Input rejected: {input_check['reason']}"
        }

    # 2. Classification
    category = classify_question(question)
    print("Selected category:", category)

    if category == "INVALID":
        return {
            "status": "rejected",
            "category": "INVALID",
            "agent": None,
            "reply": "Input rejected."
        }

    # 3. Routing
    agent = AGENTS.get(category)
    if not agent:
        return {
            "status": "unsupported",
            "category": category,
            "agent": None,
            "reply": "Unable to route this question."
        }

    print("Selected agent:", agent.name)

    # 4. Agent Execution
    result = agent.run(question)

    if not result["success"]:
        return {
            "status": "error",
            "category": category,
            "agent": agent.name,
            "reply": result["response"]
        }

    # 5. Output Guard
    output_check = OutputGuard.validate(result["response"])
    if not output_check["valid"]:
        return {
            "status": "rejected",
            "category": category,
            "agent": agent.name,
            "reply": f"Output rejected: {output_check['reason']}"
        }

    raw_text = result["response"]
    clean_reply = raw_text

    if "- Recommendation:" in raw_text:
        parts = raw_text.split("- Recommendation:")
        if len(parts) > 1:
            clean_reply = parts[1].strip()

    # Translate if target language is specified
    clean_reply = auto_detect_and_translate(question, clean_reply, target_language)

    print("\nFINAL RESPONSE")
    print(clean_reply)

    return {
        "status": "success",
        "category": category,
        "agent": agent.name,
        "language": target_language,
        "reply": clean_reply
    }