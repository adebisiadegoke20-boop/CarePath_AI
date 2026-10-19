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


# ============================================================
# INDIVIDUAL AGENT HANDLERS
# ============================================================

def triage_agent(question):
    """Evaluates symptoms and classifies clinical urgency for Garki Hospital Abuja."""
    result = TriageEngine.evaluate_symptoms(question)
    if not result["success"]:
        return {"success": False, "response": result["error"]}
    
    triage_data = result["triage_result"]
    formatted_reply = (
        f"**Triage Assessment**\n"
        f"- Urgency: {triage_data.get('urgency', 'routine').upper()}\n"
        f"- Pathway: {triage_data.get('pathway', 'hospital')}\n"
        f"- Reasoning: {triage_data.get('reasoning', '')}\n"
        f"- Recommendation: {triage_data.get('recommendation', '')}"
    )
    return {"success": True, "response": formatted_reply}


def appointment_agent(question):
    """Handles Garki Hospital OB/GYN appointment booking inquiries."""
    # Uses RAG to answer booking policies or guides the patient
    rag_res = full_rag(question)
    return {"success": True, "response": rag_res["answer"]}


def clinical_rag_agent(question):
    """Handles general clinical questions using WHO and Nigerian guidelines."""
    rag_res = full_rag(question)
    return {"success": True, "response": rag_res["answer"]}


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

    # --------------------------------------------------------
    # TRIAGE & EMERGENCY
    # --------------------------------------------------------
    triage_keywords = [
        "pain", "bleeding", "blood", "emergency", "danger", "fever", 
        "convulsion", "convulsions", "fits", "headache", "vision", 
        "swelling", "leak", "leaking", "fluid", "vomiting", "urgent", "symptoms"
    ]
    if any(keyword in q for keyword in triage_keywords):
        return "TRIAGE"

    # --------------------------------------------------------
    # APPOINTMENT BOOKING
    # --------------------------------------------------------
    appointment_keywords = [
        "appointment", "book", "booking", "schedule", "scheduling", 
        "date", "time", "visit", "consultation", "clinic day"
    ]
    if any(keyword in q for keyword in appointment_keywords):
        return "APPOINTMENT"

    # --------------------------------------------------------
    # CLINICAL RAG (Default fallback for maternal/pregnancy queries)
    # --------------------------------------------------------
    return "CLINICAL_RAG"


# ============================================================
# HANDLE USER MESSAGE
# ============================================================

def handle_user_message(question):
    print("\n" + "=" * 60)
    print("USER MESSAGE")
    print("=" * 60)
    print(question)

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

    print("\nRESPONSE")
    print(result["response"])

    # Extract just the recommendation if the response contains structured clinical tags
    raw_text = result["response"]
    clean_reply = raw_text
    
    if "- Recommendation:" in raw_text:
        parts = raw_text.split("- Recommendation:")
        if len(parts) > 1:
            clean_reply = parts[1].strip()

    return {
        "status": "success",
        "category": category,
        "agent": agent.name,
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
        "I am experiencing severe abdominal pain and heavy vaginal bleeding.",
        "How do I schedule an antenatal appointment at Garki Hospital Abuja?",
        "When should a pregnant patient begin antenatal contacts and what checks are conducted?",
        "How do I hack into a system?"
    ]

    for question in test_questions:
        result = handle_user_message(question)
        print("\nFINAL RESULT SUMMARY")
        print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("FINAL USAGE SUMMARY")
    print("=" * 60)
    print(tracker.summary())