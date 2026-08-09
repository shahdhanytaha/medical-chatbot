from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag import llm, retriever


# =========================================================
# Query Rewriting Prompt
# =========================================================

rewrite_prompt = PromptTemplate(
    input_variables=[
        "history",
        "question"
    ],

    template="""
You are a medical conversation query rewriting assistant.

Use the conversation history to understand references such as:

- it
- its
- this disease
- this condition
- that disease
- they
- them

Rewrite the current user question into a clear,
standalone medical question.

IMPORTANT:

- Do not answer the question.
- Do not add medical information.
- Do not change the meaning.
- If the question is already clear, keep it almost unchanged.

Conversation History:
{history}

Current Question:
{question}

Standalone Question:
"""
)


rewrite_chain = (
    rewrite_prompt
    | llm
    | StrOutputParser()
)


# =========================================================
# Medical Answer Prompt
# =========================================================

answer_prompt = PromptTemplate(
    input_variables=[
        "history",
        "context",
        "question"
    ],

    template="""
You are an AI Medical Assistant.

Use the conversation history to understand the user's
question and references.

Answer the user's medical question ONLY using
the provided medical context.

Do not invent information.

If the answer is not available in the medical context,
say exactly:

"I don't have enough information in my medical database."

Conversation History:
{history}

Medical Context:
{context}

Question:
{question}

Answer:
"""
)


answer_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)


# =========================================================
# Memory
# =========================================================

chat_history = []


# =========================================================
# Ask Medical Question
# =========================================================

def ask_medical_question(question):

    # -----------------------------------------------------
    # Build history
    # -----------------------------------------------------

    history = "\n".join(
        chat_history
    )


    # -----------------------------------------------------
    # Rewrite question using memory
    # -----------------------------------------------------

    standalone_question = rewrite_chain.invoke(
        {
            "history": history,
            "question": question
        }
    )


    # -----------------------------------------------------
    # Retrieve using rewritten question
    # -----------------------------------------------------

    docs = retriever.invoke(
        standalone_question
    )


    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    answer = answer_chain.invoke(
        {
            "history": history,
            "context": context,
            "question": standalone_question
        }
    )


    # -----------------------------------------------------
    # Save memory
    # -----------------------------------------------------

    chat_history.append(
        f"User: {question}"
    )

    chat_history.append(
        f"Assistant: {answer}"
    )


    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    sources = list(
        set(
            doc.metadata.get(
                "source",
                "Unknown"
            )
            for doc in docs
        )
    )


    return answer, sources