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


def translate_response(text: str, target_language_code: str) -> str:
    """
    Translates the final clinical recommendation into the patient's selected language.
    """
    if not target_language_code or target_language_code == "en" or target_language_code not in LANGUAGE_NAMES:
        return text

    target_lang_name = LANGUAGE_NAMES[target_language_code]

    prompt = f"""You are an empathetic medical translator at Garki Hospital Abuja.
Translate the following clinical guidance into clear, natural, and accessible {target_lang_name}.
Keep medical terms easy to understand for the patient. Return ONLY the translated text without extra conversational filler.

Text to translate:
"{text}"
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation Error ({target_lang_name}):", str(e))
        return text


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
    answer = rag_res.get("answer", "")
    
    if not answer or "don't have enough information" in answer.lower():
        answer = "To inquire about or schedule an OB/GYN appointment at Garki Hospital Abuja, please visit our ANC clinic reception desk Monday through Friday (8:00 AM – 4:00 PM)."
        
    return {"success": True, "response": answer}


def clinical_rag_agent(question):
    """Handles general clinical questions using WHO and Nigerian guidelines."""
    rag_res = full_rag(question)
    answer = rag_res.get("answer", "")
    
    if not answer or "don't have enough information" in answer.lower():
        answer = "For general pregnancy and maternal health inquiries, please consult the clinical team at Garki Hospital Abuja OB/GYN Department during routine ANC visits."
        
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
            result = triage_agent(question)
        elif self.name == "Appointment Booking Agent":
            result = appointment_agent(question)
        elif self.name == "Clinical Knowledge Agent":
            result = clinical_rag_agent(question)
        else:
            return {
                "success": False,
                "response": "Unknown agent."
            }
        return result


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
        "swelling", "leak", "leaking", "fluid", "vomiting", "urgent", "symptoms"
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

    # --------------------------------------------------------
    # INPUT GUARD
    # --------------------------------------------------------
    input_check = InputGuard.validate(question)
    if not input_check["valid"]:
        print("\nINPUT GUARD REJECTION:")
        print(input_check["reason"])
        return {
            "status": "rejected",
            "category": "INVALID",
            "agent": None,
            "reply": f"Input rejected: {input_check['reason']}"
        }

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------
    print("\nSTEP 1: CLASSIFICATION")
    category = classify_question(question)
    print("Selected category:", category)

    if category == "INVALID":
        return {
            "status": "rejected",
            "category": "INVALID",
            "agent": None,
            "reply": "Input rejected."
        }

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------
    agent = AGENTS.get(category)
    if not agent:
        return {
            "status": "unsupported",
            "category": category,
            "agent": None,
            "reply": "Unable to route this question."
        }

    print("\nSTEP 2: ROUTING")
    print("Selected agent:", agent.name)

    # --------------------------------------------------------
    # AGENT EXECUTION
    # --------------------------------------------------------
    print("\nSTEP 3: AGENT EXECUTION")
    result = agent.run(question)

    if not result["success"]:
        print("\nAGENT ERROR:")
        print(result["response"])
        return {
            "status": "error",
            "category": category,
            "agent": agent.name,
            "reply": result["response"]
        }

    # --------------------------------------------------------
    # OUTPUT GUARD
    # --------------------------------------------------------
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

    # Translate response to the selected patient language if required
    if target_language and target_language != "en":
        clean_reply = translate_response(clean_reply, target_language)

    print("\nFINAL RESPONSE")
    print(clean_reply)

    return {
        "status": "success",
        "category": category,
        "agent": agent.name,
        "language": target_language,
        "reply": clean_reply
    }


# ============================================================
# TEST QUESTIONS
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CAREPATH MULTI-AGENT SYSTEM TEST")
    print("=" * 60)

    test_questions = [
        ("I am experiencing severe abdominal pain and heavy vaginal bleeding.", "en"),
        ("When should a pregnant patient begin antenatal contacts and what checks are conducted?", "ha"),
        ("Is my morning sickness and nausea normal during pregnancy?", "yo"),
        ("How do I hack into a system?", "en")
    ]

    for question, lang in test_questions:
        result = handle_user_message(question, target_language=lang)
        print("\nFINAL RESULT SUMMARY")
        print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("FINAL USAGE SUMMARY")
    print("=" * 60)
    print(tracker.summary())