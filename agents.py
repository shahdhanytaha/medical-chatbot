import os
from typing import TypedDict

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing.")


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key
)


class AgentState(TypedDict):
    question: str
    context: str
    history: str
    report: str
    route: str
    answer: str


# ==========================================
# Supervisor Agent
# ==========================================

def supervisor_agent(state):

    question = state["question"]

    prompt = f"""
You are the supervisor of a medical AI system.

Decide which agent should handle the request.

Choose ONLY:

RAG

or

SUMMARY

Use RAG for medical knowledge questions.

Use SUMMARY for requests to summarize,
analyze, or explain a medical report.

User request:
{question}

Return ONLY RAG or SUMMARY.
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


# ==========================================
# RAG Agent
# ==========================================

def rag_agent(state):

    from chatbot import ask_medical_question

    question = state["question"]

    answer, sources = ask_medical_question(
        question
    )

    return {
        "answer": answer
    }


# ==========================================
# Summary Agent
# ==========================================

def summary_agent(state):

    from summarizer import summarize_report

    report = state["report"]

    answer = summarize_report(
        report
    )

    return {
        "answer": answer
    }


# ==========================================
# Router
# ==========================================

def route_agent(state):

    if state["route"] == "SUMMARY":
        return "summary"

    return "rag"


# ==========================================
# Build Graph
# ==========================================

graph = StateGraph(AgentState)


graph.add_node(
    "supervisor",
    supervisor_agent
)

graph.add_node(
    "rag",
    rag_agent
)

graph.add_node(
    "summary",
    summary_agent
)


graph.add_edge(
    START,
    "supervisor"
)


graph.add_conditional_edges(
    "supervisor",
    route_agent,
    {
        "rag": "rag",
        "summary": "summary"
    }
)


graph.add_edge(
    "rag",
    END
)

graph.add_edge(
    "summary",
    END
)


agent_graph = graph.compile()


# ==========================================
# Main Function
# ==========================================

def run_agents(
    question,
    context="",
    history="",
    report=""
):

    result = agent_graph.invoke({

        "question": question,

        "context": context,

        "history": history,

        "report": report,

        "route": "",

        "answer": ""
    })

    return result["answer"]