
import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from core.llm import get_llm

load_dotenv() 


# def get_llm():
#     """Initialize the Mistral language model."""

#     api_key = os.getenv("MISTRAL_API_KEY")

#     if not api_key:
#         raise ValueError("MISTRAL_API_KEY is missing.")

#     return ChatMistralAI(
#         model="mistral-small-latest",
#         mistral_api_key=api_key,
#         temperature=0.2
#     )


def build_chain(system_prompt: str):
    """Build a reusable extraction chain."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    return prompt | get_llm() | StrOutputParser()


def extract_action_items(transcript: str) -> str:
    """Extract action items, owners and deadlines."""

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    chain = build_chain("""
    Extract the action items from this meeting transcript.

    Include:
    - Task
    - Owner, if mentioned
    - Deadline, if mentioned

    Do not invent information.
    If no action items are found, say 'No action items found.'
    """)

    return chain.invoke({"text": transcript})


def extract_key_decisions(transcript: str) -> str:
    """Extract finalized meeting decisions."""

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    chain = build_chain("""
    Extract only the important decisions finalized during the meeting.

    Exclude suggestions and unresolved proposals.
    Do not invent information.
    If no decisions are found, say 'No key decisions found.'
    """)

    return chain.invoke({"text": transcript})


def extract_questions(transcript: str) -> str:
    """Extract unresolved questions and follow-up topics."""

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    chain = build_chain("""
    Extract unresolved questions and topics requiring follow-up.

    Exclude questions that were already answered.
    Do not invent information.
    If none are found, say 'No open questions found.'
    """)

    return chain.invoke({"text": transcript})
