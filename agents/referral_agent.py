from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def run_referral_agent(context: str, severity: str) -> str:
    """Suggest doctor referral based on report and severity level."""
    try:
        
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"  
        )

        prompt = f"""
        You are a medical referral assistant.
        Based on the medical report and severity level, suggest what the patient should do.

        Medical Report Context:
        {context}

        Severity Level: {severity}

        Provide:
        1. Which type of doctor to visit (if needed)
        2. How urgent is the visit (immediately/within a week/routine checkup)
        3. Any home care tips in the meantime

        Keep it simple, clear and helpful.
        Always end with: "This is not medical advice. Please consult a doctor."
        """

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"[Referral Agent Error] {str(e)}"