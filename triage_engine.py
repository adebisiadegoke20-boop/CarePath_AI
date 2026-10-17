import os
import json
from dotenv import load_dotenv
from groq import Groq
from carepath_rag_engine import rag_search
from guard import InputGuard, OutputGuard, tracker

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("Triage Engine Initialized")

class TriageEngine:
    """Evaluates pregnancy symptoms against clinical guidelines using Groq and RAG."""
    
    TRIAGE_SYSTEM_PROMPT = (
        "You are an expert clinical triage assistant for the Obstetrics & Gynecology (OB/GYN) "
        "Department at Garki Hospital Abuja, utilizing WHO and Nigerian National Health Guidelines. "
        "Analyze the patient symptoms and reference context to determine the clinical urgency.\n\n"
        "Classify urgency into EXACTLY one of these categories:\n"
        "1. 'emergency' (Life-threatening: heavy vaginal bleeding, severe abdominal pain, high fever, "
        "fits or convulsions, fluid leakage, severe headache with blurred vision)\n"
        "2. 'urgent' (Requires medical evaluation within 24 hours)\n"
        "3. 'routine' (Manageable via scheduled ANC contacts or routine consultation)\n\n"
        "Determine the care pathway: 'hospital' (Garki Hospital Abuja OB/GYN ER or clinic), 'phc', or 'selfcare'.\n\n"
        "You MUST return a valid JSON object ONLY with the following keys:\n"
        "- \"urgency\": string\n"
        "- \"pathway\": string\n"
        "- \"reasoning\": string\n"
        "- \"recommendation\": string"
    )

    @classmethod
    def evaluate_symptoms(cls, symptom_text: str) -> dict:
        """Runs input validation, RAG context injection, Groq LLM inference, and JSON output validation."""
        
        # Layer 1: Input Guardrail Validation
        input_check = InputGuard.validate(symptom_text)
        if not input_check["valid"]:
            return {
                "success": False,
                "error": input_check["reason"],
                "triage_result": None
            }

        # Step 2: Retrieve relevant clinical text via RAG search
        docs = rag_search(symptom_text, top_k=3)
        context_text = "\n---\n".join([d.get("content", "") for d in docs]) if docs else "Standard WHO and Garki Hospital guidelines apply."
        
        user_content = f"Reference Guidelines:\n{context_text}\n\nPatient Symptoms/Query: {symptom_text}"

        # Layer 2: Groq LLM Call enforcing JSON response format
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": cls.TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=600,
                response_format={"type": "json_object"}
            )

            tracker.track(response.usage)
            ai_text = response.choices[0].message.content

            # Layer 3: Output Guardrail Validation (Ensures valid JSON & clean content)
            output_check = OutputGuard.validate(ai_text, expected_format="json")
            if not output_check["valid"]:
                return {
                    "success": False,
                    "error": "Response failed validation: " + output_check["reason"],
                    "triage_result": None
                }

            triage_data = json.loads(ai_text)

            # Append Garki Hospital Abuja clinical notice to recommendation
            if "recommendation" in triage_data:
                triage_data["recommendation"] += (
                    " [Clinical Notice: For emergencies, report directly to the "
                    "Obstetrics & Gynecology (OB/GYN) Department at Garki Hospital Abuja.]"
                )

            return {
                "success": True,
                "error": None,
                "triage_result": triage_data,
                "tokens": response.usage.total_tokens,
                "tracker": tracker.summary()
            }

        except Exception as e:
            return {
                "success": False,
                "error": "API Error: " + str(e),
                "triage_result": None
            }