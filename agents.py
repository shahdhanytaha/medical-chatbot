import os
from typing import TypedDict

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END


# =====================================
# Environment
# =====================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing.")


# =====================================
# LLM
# =====================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key
)


# =====================================
# State
# =====================================

class AgentState(TypedDict):
    question: str
    context: str
    report: str
    route: str
    answer: str


# =====================================
# Supervisor Agent
# =====================================

def supervisor_agent(state: AgentState):

    question = state["question"].strip()

    # -----------------------------
    # Greeting detection
    # -----------------------------

    greeting_words = {
        "hi",
        "hello",
        "hey",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening"
    }

    if question.lower() in greeting_words:

        return {
            "route": "GREETING"
        }

    # -----------------------------
    # Supervisor prompt
    # -----------------------------

    prompt = f"""
You are a supervisor for a medical AI system.

Decide which agent should handle the user request.

Choose ONLY one:

RAG
SUMMARY

Use RAG when the user asks a medical knowledge question.

Use SUMMARY when the user asks to summarize,
analyze, or explain a medical report.

User request:
{question}

Return ONLY:

RAG

or:

SUMMARY
"""

    result = llm.invoke(prompt)

    route = result.content.strip().upper()

    if "SUMMARY" in route:
        route = "SUMMARY"
    else:
        route = "RAG"

    return {
        "route": route
    }


# =====================================
# RAG Agent
# =====================================

def rag_agent(state: AgentState):

    context = state["context"]
    question = state["question"]

    prompt = PromptTemplate(
        input_variables=[
            "context",
            "question"
        ],

        template="""
You are a Medical RAG Agent.

Answer ONLY using the provided medical context.

Do not invent information.

If the answer is not found in the context, say:

"I don't have enough information in my medical database."

Medical Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return {
        "answer": answer
    }


# =====================================
# Summarizer Agent
# =====================================

def summarizer_agent(state: AgentState):

    report = state["report"]

    if not report.strip():

        return {
            "answer": "The uploaded report does not contain readable text."
        }

    prompt = """
You are a Medical Report Summarizer Agent.

Summarize ONLY the information contained
in the medical report.

Do not diagnose the patient.

Do not invent information.

Organize the response into:

## Summary

## Key Findings

## Important Values

## Medications Mentioned

## Recommendations or Follow-up Mentioned

Medical Report:

""" + report

    result = llm.invoke(prompt)

    return {
        "answer": result.content
    }


# =====================================
# Greeting Agent
# =====================================

def greeting_agent(state: AgentState):

    return {
        "answer": (
            "Hello! 👋 I am your Medical AI Assistant. "
            "How can I help you?"
        )
    }


# =====================================
# Router
# =====================================

def route_agent(state: AgentState):

    if state["route"] == "SUMMARY":
        return "summarizer"

    if state["route"] == "GREETING":
        return "greeting"

    return "rag"


# =====================================
# Build Graph
# =====================================

graph = StateGraph(AgentState)


# -----------------------------
# Add Agents
# -----------------------------

graph.add_node(
    "supervisor",
    supervisor_agent
)

graph.add_node(
    "rag",
    rag_agent
)

graph.add_node(
    "summarizer",
    summarizer_agent
)

graph.add_node(
    "greeting",
    greeting_agent
)


# -----------------------------
# Start
# -----------------------------

graph.add_edge(
    START,
    "supervisor"
)


# -----------------------------
# Supervisor Routing
# -----------------------------

graph.add_conditional_edges(
    "supervisor",
    route_agent,
    {
        "rag": "rag",
        "summarizer": "summarizer",
        "greeting": "greeting"
    }
)


# -----------------------------
# End
# -----------------------------

graph.add_edge(
    "rag",
    END
)

graph.add_edge(
    "summarizer",
    END
)

graph.add_edge(
    "greeting",
    END
)


# =====================================
# Compile Graph
# =====================================

agent_graph = graph.compile()


# =====================================
# Run Agents
# =====================================

def run_agents(
    question,
    context="",
    report=""
):

    result = agent_graph.invoke(
        {
            "question": question,
            "context": context,
            "report": report,
            "route": "",
            "answer": ""
        }
    )

    return result["answer"]