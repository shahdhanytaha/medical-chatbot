import os
import tempfile

import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from chatbot import ask_medical_question
from agents import run_agents


# =========================================================
# Environment
# =========================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. Please check your .env file."
    )
    st.stop()


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Medical AI Chatbot",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "report_text" not in st.session_state:
    st.session_state.report_text = ""

if "report_id" not in st.session_state:
    st.session_state.report_id = None


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.title("🩺 Medical AI")

    st.markdown("---")

    # -----------------------------------------------------
    # Clear Chat
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    # -----------------------------------------------------
    # Medical Knowledge PDF
    # -----------------------------------------------------

    st.subheader("📚 Medical Knowledge PDF")

    knowledge_pdf = st.file_uploader(
        "Upload Medical Knowledge PDF",
        type=["pdf"],
        key="knowledge_pdf"
    )

    if knowledge_pdf is not None:

        st.success(
            f"Selected: {knowledge_pdf.name}"
        )

        st.info(
            "Knowledge PDF upload is available. "
            "Your existing ChromaDB remains active."
        )

    st.markdown("---")

    # -----------------------------------------------------
    # Medical Report
    # -----------------------------------------------------

    st.subheader("📄 Medical Report")

    uploaded_report = st.file_uploader(
        "Upload Medical Report",
        type=["pdf"],
        key="medical_report"
    )

    st.markdown("---")

    # -----------------------------------------------------
    # Agents
    # -----------------------------------------------------

    st.subheader("🤖 Multi-Agent System")

    st.write("Supervisor Agent")
    st.write("↳ RAG Agent")
    st.write("↳ Summary Agent")


# =========================================================
# Medical Report Processing
# =========================================================

if uploaded_report is not None:

    report_id = (
        uploaded_report.name,
        uploaded_report.size
    )

    # Process only when a new file is uploaded
    if st.session_state.report_id != report_id:

        temp_path = None

        try:

            with st.spinner(
                "Reading medical report..."
            ):

                # -----------------------------------------
                # Save PDF temporarily
                # -----------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_report.getbuffer()
                    )

                    temp_path = temp_file.name


                # -----------------------------------------
                # Load PDF
                # -----------------------------------------

                loader = PyPDFLoader(
                    temp_path
                )

                documents = loader.load()


                # -----------------------------------------
                # Extract text
                # -----------------------------------------

                report_pages = []

                for doc in documents:

                    text = doc.page_content.strip()

                    if text:
                        report_pages.append(text)


                report_text = "\n\n".join(
                    report_pages
                )


                # -----------------------------------------
                # Save report
                # -----------------------------------------

                if report_text:

                    st.session_state.report_text = (
                        report_text
                    )

                    st.session_state.report_id = (
                        report_id
                    )

                    st.sidebar.success(
                        "✅ Medical report loaded!"
                    )

                else:

                    st.session_state.report_text = ""

                    st.sidebar.warning(
                        "⚠️ This PDF contains no readable text."
                    )


        except Exception as e:

            st.sidebar.error(
                f"Report error: {e}"
            )


        finally:

            if (
                temp_path
                and os.path.exists(temp_path)
            ):

                os.remove(temp_path)


# =========================================================
# Main Header
# =========================================================

st.title("🩺 Medical AI Chatbot")

st.caption(
    "Medical RAG • Conversation Memory • Multi-Agent System"
)


# =========================================================
# Medical Report Section
# =========================================================

if st.session_state.report_text:

    st.markdown("---")

    st.header("📄 Uploaded Medical Report")

    with st.expander(
        "View report"
    ):

        st.text(
            st.session_state.report_text
        )


    # -----------------------------------------------------
    # Summary Button
    # -----------------------------------------------------

    if st.button(
        "✨ Summarize Medical Report",
        use_container_width=True
    ):

        with st.spinner(
            "Supervisor Agent → Summary Agent..."
        ):

            try:

                summary = run_agents(

                    question=(
                        "Summarize this medical report"
                    ),

                    context="",

                    history="",

                    report=(
                        st.session_state.report_text
                    )
                )


                st.subheader(
                    "📋 Medical Report Summary"
                )

                st.markdown(
                    summary
                )


            except Exception as e:

                st.error(
                    f"Summary error: {e}"
                )


# =========================================================
# Display Previous Messages
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # Show sources for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"- {source}"
                    )


# =========================================================
# Chat Input
# =========================================================

question = st.chat_input(
    "Ask a medical question..."
)


# =========================================================
# Handle User Question
# =========================================================

if question:

    # -----------------------------------------------------
    # Show User Message
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # -----------------------------------------------------
    # Save User Message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # -----------------------------------------------------
    # Ask Medical Chatbot
    #
    # chatbot.py handles:
    # - conversation memory
    # - query rewriting
    # - retrieval
    # - answer generation
    # - sources
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Supervisor + Medical Agents are working..."
        ):

            try:

                answer, sources = ask_medical_question(
                    question
                )

                st.markdown(
                    answer
                )


                # -----------------------------------------
                # Sources
                # -----------------------------------------

                with st.expander(
                    "📚 Sources"
                ):

                    if sources:

                        for source in sources:

                            st.write(
                                f"- {source}"
                            )

                    else:

                        st.write(
                            "No sources found."
                        )


                # -----------------------------------------
                # Save Assistant Message
                # -----------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )


            except Exception as e:

                error_message = (
                    f"Agent error: {e}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )