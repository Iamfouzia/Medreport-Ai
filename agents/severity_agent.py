from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def run_severity_agent(context: str) -> str:
    """Check if medical report contains serious or urgent values."""
    try:
        
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"  
        )

        prompt = f"""
        You are a medical severity checker.
        Read the following medical report and identify any abnormal or dangerous values.

        Context:
        {context}

        Return one of these severity levels:
        - LOW: All values are normal
        - MEDIUM: Some values are slightly abnormal
        - HIGH: Critical values found, immediate attention needed

        Also explain why in 2-3 simple sentences.
        """

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"[Severity Agent Error] {str(e)}"