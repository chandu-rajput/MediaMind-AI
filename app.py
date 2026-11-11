
import os
import tempfile
from pathlib import Path

import streamlit as st

from main import run_pipeline
from core.rag_engine import ask_question


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ AI Meeting Assistant")
st.caption("Turn meeting recordings into summaries, insights, and actionable notes.")


# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# Sidebar - Input Configuration
# --------------------------------------------------

with st.sidebar:

    st.header("Meeting Input")

    input_type = st.radio(
        "Choose input method",
        ["YouTube URL", "Upload Audio/Video"]
    )

    source_url = ""

    uploaded_file = None

    if input_type == "YouTube URL":

        source_url = st.text_input(
            "YouTube Video URL",
            placeholder="Paste YouTube URL here..."
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload meeting recording",
            type=[
                "mp3",
                "wav",
                "m4a",
                "mp4",
                "mkv",
                "mov",
                "webm"
            ]
        )

    language = st.selectbox(
        "Meeting Language",
        ["English", "Hindi", "Hinglish"]
    )

    process_button = st.button(
        "Process Meeting",
        use_container_width=True,
        type="primary"
    )

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []


# --------------------------------------------------
# Process Meeting
# --------------------------------------------------

if process_button:

    source = None
    temp_path = None

    if input_type == "YouTube URL":

        if not source_url.strip():
            st.error("Please enter a YouTube URL.")

        else:
            source = source_url.strip()

    else:

        if uploaded_file is None:
            st.error("Please upload an audio or video file.")

        else:

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_file.name).suffix
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                source = temp_path

            except Exception as e:
                st.error(f"File upload error: {e}")

    if source:

        try:

            with st.spinner(
                "Processing your meeting... This may take some time."
            ):

                result = run_pipeline(
                    source,
                    language=language
                )

                st.session_state.result = result
                st.session_state.chat_history = []

            st.success("Meeting processed successfully!")

        except Exception as e:

            st.error(f"Error processing meeting: {e}")

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# --------------------------------------------------
# Display Meeting Results
# --------------------------------------------------

if st.session_state.result:

    result = st.session_state.result

    st.header(result["title"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Summary",
            "Action Items",
            "Key Decisions",
            "Open Questions",
            "Transcript"
        ]
    )

    with tab1:

        st.subheader("Meeting Summary")
        st.write(result["summary"])

    with tab2:

        st.subheader("Action Items")
        st.write(result["action_items"])

    with tab3:

        st.subheader("Key Decisions")
        st.write(result["key_decisions"])

    with tab4:

        st.subheader("Open Questions")
        st.write(result["open_questions"])

    with tab5:

        st.subheader("Full Transcript")

        st.text_area(
            "Meeting Transcript",
            value=result["transcript"],
            height=400,
            disabled=True
        )


    # --------------------------------------------------
    # Download Meeting Notes
    # --------------------------------------------------

    meeting_notes = f"""
AI MEETING ASSISTANT
====================

Title:
{result['title']}

SUMMARY
-------
{result['summary']}

ACTION ITEMS
------------
{result['action_items']}

KEY DECISIONS
-------------
{result['key_decisions']}

OPEN QUESTIONS
--------------
{result['open_questions']}

FULL TRANSCRIPT
---------------
{result['transcript']}
"""

    st.download_button(
        label="Download Meeting Notes (TXT)",
        data=meeting_notes,
        file_name="meeting_notes.txt",
        mime="text/plain"
    )


    # --------------------------------------------------
    # RAG Chat - Ask Questions About Meeting
    # --------------------------------------------------

    st.divider()

    st.header("💬 Chat with Your Meeting")

    st.caption(
        "Ask questions about the meeting transcript."
    )

    # Display previous chat messages

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User question input

    question = st.chat_input(
        "Ask something about this meeting..."
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        try:

            with st.chat_message("assistant"):

                with st.spinner("Finding the answer..."):

                    answer = ask_question(
                        result["rag_chain"],
                        question
                    )

                    st.write(answer)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            st.error(f"Error generating answer: {e}")

else:

    st.info(
        "Enter a YouTube URL or upload a meeting recording "
        "from the sidebar to get started."
    )
