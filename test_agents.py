from agents import run_agents


# Test RAG Agent

answer = run_agents(
    question="What is diabetes?"
)

print("\nRAG RESULT:\n")
print(answer)


# Test Summary Agent

report = """
Blood Glucose: 180 mg/dL
Blood Pressure: 145/90 mmHg
Medication: Metformin 500 mg
Follow-up: Repeat blood glucose test.
"""


answer = run_agents(
    question="Summarize this medical report",
    report=report
)

print("\nSUMMARY RESULT:\n")
print(answer)