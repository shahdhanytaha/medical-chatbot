from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag import llm, retriever


prompt = PromptTemplate(
    input_variables=[
        "history",
        "context",
        "question"
    ],

    template="""
You are an AI Medical Assistant.

Use the conversation history only to understand references such as:
it, its, this disease, that condition.

Answer the user's medical question ONLY using the provided medical context.

Do not invent information.

If the answer is not available in the medical context, say exactly:

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

chain = prompt | llm | StrOutputParser()

chat_history = []


def ask_medical_question(question):

    history = "\n".join(chat_history[-10:])

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    answer = chain.invoke(
        {
            "history": history,
            "context": context,
            "question": question
        }
    )

    chat_history.append(
        f"User: {question}"
    )

    chat_history.append(
        f"Assistant: {answer}"
    )

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