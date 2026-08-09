import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =====================================
# Load .env
# =====================================

load_dotenv()


# =====================================
# Check API Key
# =====================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Please check your .env file."
    )


# =====================================
# Load LLM
# =====================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key
)


# =====================================
# Prompt
# =====================================

summary_prompt = PromptTemplate(
    input_variables=["report"],
    template="""
You are a medical report summarization assistant.

Summarize the medical report using ONLY the information
provided in the report.

Rules:
- Do not diagnose the patient.
- Do not invent information.
- Do not add medical facts that are not present.
- Keep the summary clear and easy to understand.
- If information is not present, write:
  "Not mentioned in the report."

Organize the answer into:

## Summary

## Key Findings

## Important Values

## Medications Mentioned

## Recommendations or Follow-up Mentioned

Medical Report:
{report}

Summary:
"""
)


# =====================================
# Chain
# =====================================

summary_chain = (
    summary_prompt
    | llm
    | StrOutputParser()
)


# =====================================
# Function
# =====================================

def summarize_report(report_text):

    if not report_text.strip():
        return "The uploaded report does not contain readable text."

    return summary_chain.invoke(
        {
            "report": report_text
        }
    )