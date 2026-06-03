from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def run_report_agent(context: str, question: str) -> str:
    """Summarize medical report and answer user question."""
    try:
    
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

        prompt = f"""
        You are a medical report assistant.
        Based on the following medical report context, answer the question clearly.

        Context:
        {context}

        Question: {question}

        Give a clear, simple answer in plain English.
        Do not make assumptions beyond the report.
        """

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"[Report Agent Error] {str(e)}"