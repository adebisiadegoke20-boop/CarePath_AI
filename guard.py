import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ============================================================
# INPUT GUARDRALL
# ============================================================
class InputGuard:
    """Validates and sanitizes user input before sending it to the LLM."""
    PROHIBITED_TOPICS = ["hack", "exploit", "illegal", "malware", "password", "drop table"]
    MAX_LENGTH = 2000
    MIN_LENGTH = 3  # Adjusted for clinical phrases like "bleeding" or "pain"

    @classmethod
    def validate(cls, user_input):
        if not user_input or len(user_input.strip()) < cls.MIN_LENGTH:
            return {"valid": False, "reason": "Input too short. Minimum " + str(cls.MIN_LENGTH) + " characters required."}
        if len(user_input) > cls.MAX_LENGTH:
            return {"valid": False, "reason": "Input too long. Maximum " + str(cls.MAX_LENGTH) + " characters allowed."}
        
        lower = user_input.lower()
        for topic in cls.PROHIBITED_TOPICS:
            if topic in lower:
                return {"valid": False, "reason": "Prohibited topic detected: " + topic}
        return {"valid": True, "reason": None}


# ============================================================
# OUTPUT GUARDRAIL
# ============================================================
class OutputGuard:
    """Validates LLM output before showing it to the user."""
    PROHIBITED_PHRASES = ["I guarantee", "I promise", "100%", "definitely safe", "no risk"]
    MAX_RESPONSE_LENGTH = 8000  # Increased to allow detailed WHO clinical outputs

    @classmethod
    def validate(cls, ai_response, expected_format=None):
        if not ai_response:
            return {"valid": False, "reason": "Empty response"}
        if len(ai_response) > cls.MAX_RESPONSE_LENGTH:
            return {"valid": False, "reason": "Response too long: " + str(len(ai_response)) + " characters"}
        
        for phrase in cls.PROHIBITED_PHRASES:
            if phrase.lower() in ai_response.lower():
                return {"valid": False, "reason": "Prohibited phrase found: " + phrase}
                
        if expected_format == "json":
            try:
                json.loads(ai_response)
            except json.JSONDecodeError:
                return {"valid": False, "reason": "Response is not valid JSON"}
                
        return {"valid": True, "reason": None}
# ============================================================
# COST & TOKEN TRACKER
# ============================================================
class CostTracker:
    def __init__(self, daily_budget=1.00):
        self.daily_budget = daily_budget
        self.total_tokens = 0
        self.calls = 0

    def track(self, usage):
        if usage:
            self.total_tokens += usage.total_tokens or 0
        self.calls += 1

    def summary(self):
        return {
            "calls": self.calls,
            "tokens": self.total_tokens,
            "budget_status": "Groq free tier / optimized usage"
        }

tracker = CostTracker()


# ============================================================
# SAFE AI EXECUTION WRAPPER
# ============================================================
def safe_ai_call(user_input, system_prompt):
    """Executes input validation, Groq inference, and output validation."""
    # Layer 1: Input Guardrail
    input_check = InputGuard.validate(user_input)
    if not input_check["valid"]:
        return {
            "success": False,
            "error": input_check["reason"],
            "response": None
        }

    # Layer 2: LLM Call via Groq
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,
            max_tokens=800
        )

        tracker.track(response.usage)
        ai_text = response.choices[0].message.content

        # Layer 3: Output Guardrail
        output_check = OutputGuard.validate(ai_text)
        if not output_check["valid"]:
            return {
                "success": False,
                "error": "Response failed validation: " + output_check["reason"],
                "response": None
            }

        # Append official CarePath Garki Hospital Disclaimer
        disclaimer = (
            "\n\n[Clinical Notice: CarePath provides preliminary triage guidance based on "
            "WHO and Nigerian National Guidelines. For urgent medical concerns, report directly "
            "to the Obstetrics & Gynecology (OB/GYN) Department at Garki Hospital Abuja.]"
        )
        final_response = ai_text + disclaimer

        return {
            "success": True,
            "response": final_response,
            "tokens": response.usage.total_tokens,
            "tracker": tracker.summary()
        }

    except Exception as e:
        return {
            "success": False,
            "error": "API Error: " + str(e),
            "response": None
        }


# ============================================================
# MODULE SELF-TEST
# ============================================================
if __name__ == "__main__":
    print("Guard System Initialized Successfully")
    print("-" * 40)
    
    # Test Input Guard
    test_input = "What should I know about antenatal appointments?"
    validation = InputGuard.validate(test_input)
    print(f"Input Validation Test ('{test_input}'): {validation}")
    
    # Test Output Guard
    test_output = "I guarantee you will have a 100% safe pregnancy."
    out_validation = OutputGuard.validate(test_output)
    print(f"Output Validation Test (Prohibited Phrase): {out_validation['reason']}")