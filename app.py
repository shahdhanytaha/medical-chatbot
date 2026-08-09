import os
import tempfile

import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents import run_agents
from rag import retriever, vector_store


# =====================================
# Environment
# =====================================

load_dotenv()


# =====================================
# Page Config
# =====================================

st.set_page_config(
    page_title="Medical AI Chatbot",
    page_icon="🩺",
    layout="wide"
)


# =====================================
# Session State
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "report_text" not in st.session_state:
    st.session_state.report_text = ""

if "knowledge_loaded" not in st.session_state:
    st.session_state.knowledge_loaded = set()


# =====================================
# Header
# =====================================

st.title("🩺 Medical AI Chatbot")

st.caption(
    "Medical RAG • Conversation Memory • Multi-Agent System"
)


# =====================================
# Sidebar
# =====================================

with st.sidebar:

    st.title("🩺 Medical AI")

    st.markdown("---")


    # =================================
    # Clear Chat
    # =================================

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


    st.markdown("---")


    # =================================
    # Medical Knowledge PDF
    # =================================

    st.subheader("📚 Medical Knowledge PDF")

    knowledge_pdf = st.file_uploader(
        "Upload Medical Knowledge PDF",
        type=["pdf"],
        key="knowledge_pdf"
    )

    if knowledge_pdf is not None:

        file_id = (
            knowledge_pdf.name,
            knowledge_pdf.size
        )

        if file_id not in st.session_state.knowledge_loaded:

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:

                    tmp.write(
                        knowledge_pdf.getvalue()
                    )

                    pdf_path = tmp.name


                loader = PyPDFLoader(
                    pdf_path
                )

                documents = loader.load()


                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100
                )

                chunks = splitter.split_documents(
                    documents
                )


                vector_store.add_documents(
                    chunks
                )


                st.session_state.knowledge_loaded.add(
                    file_id
                )


                st.success(
                    f"✅ {knowledge_pdf.name} added to medical database."
                )


            except Exception as e:

                st.error(
                    f"Error loading PDF: {e}"
                )


    st.caption("200MB per file • PDF")


    st.markdown("---")


    # =================================
    # Medical Report
    # =================================

    st.subheader("📄 Medical Report")

    report_pdf = st.file_uploader(
        "Upload Medical Report",
        type=["pdf"],
        key="report_pdf"
    )


    if report_pdf is not None:

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(
                    report_pdf.getvalue()
                )

                report_path = tmp.name


            loader = PyPDFLoader(
                report_path
            )

            report_documents = loader.load()


            st.session_state.report_text = "\n\n".join(
                doc.page_content
                for doc in report_documents
            )


            st.success(
                "✅ Medical report uploaded."
            )


        except Exception as e:

            st.error(
                f"Error reading report: {e}"
            )


    st.caption("200MB per file • PDF")


    # =================================
    # Summarize Report
    # =================================

    if st.session_state.report_text:

        if st.button(
            "📋 Summarize Medical Report",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing medical report..."
            ):

                summary = run_agents(
                    question="Summarize this medical report",
                    context="",
                    report=st.session_state.report_text
                )


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": summary,
                    "sources": []
                }
            )


            st.rerun()


# =====================================
# Display Chat History
# =====================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        if message.get("sources"):

            with st.expander("📚 Sources"):

                for source in message["sources"]:

                    st.write(
                        f"- {source}"
                    )


# =====================================
# Chat Input
# =====================================

question = st.chat_input(
    "Ask a medical question..."
)


if question:

    # =================================
    # Show User Message
    # =================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "sources": []
        }
    )


    with st.chat_message("user"):

        st.markdown(
            question
        )


    # =================================
    # Handle Question
    # =================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            # -----------------------------
            # Retrieve medical context
            # -----------------------------

            docs = retriever.invoke(
                question
            )


            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )


            # -----------------------------
            # Sources
            # -----------------------------

            sources = list(
                dict.fromkeys(
                    doc.metadata.get(
                        "source",
                        "Unknown"
                    )
                    for doc in docs
                )
            )


            # -----------------------------
            # Conversation History
            # -----------------------------

            history = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages[-10:]
            )


            # -----------------------------
            # Multi-Agent System
            # -----------------------------

            answer = run_agents(
                question=question,
                context=context,
                report=st.session_state.report_text
            )


            # -----------------------------
            # Display Answer
            # -----------------------------

            st.markdown(
                answer
            )


            # -----------------------------
            # Display Sources
            # -----------------------------

            if sources:

                with st.expander("📚 Sources"):

                    for source in sources:

                        st.write(
                            f"- {source}"
                        )


    # =================================
    # Save Assistant Message
    # =================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )