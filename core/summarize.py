
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.llm import get_llm

load_dotenv()
# --------------------------------------------------
# 1. Initialize LLM
# --------------------------------------------------

# def get_llm():
#     """Initialize and return the Mistral language model."""

#     api_key = os.getenv("MISTRAL_API_KEY")

#     if not api_key:
#         raise ValueError("MISTRAL_API_KEY is missing from environment variables.")

#     return ChatMistralAI(
#         model="mistral-small-latest",
#         mistral_api_key=api_key,
#         temperature=0.3
#     )


# --------------------------------------------------
# 2. Split Transcript
# --------------------------------------------------

def split_transcript(transcript: str) -> list[str]:
    """Split a long transcript into smaller overlapping chunks."""

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


# --------------------------------------------------
# 3. Summarize Transcript (Map-Reduce)
# --------------------------------------------------

def summarize(transcript: str) -> str:
    """
    Generate a professional meeting summary using
    the Map-Reduce summarization approach.
    """

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    llm = get_llm()

    # ---------------- MAP STAGE ----------------

    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting summarizer.

            Summarize the given portion of a meeting transcript
            concisely while preserving important information.

            Focus on:
            - Main discussion points
            - Important updates
            - Decisions made
            - Action items and responsibilities
            - Deadlines, if mentioned

            Do not invent information or assumptions.
            Preserve names, numbers and important details.
            """
        ),
        ("human", "{text}")
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    # Split transcript into chunks
    chunks = split_transcript(transcript)

    # Generate a summary for each chunk
    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    # Combine all partial summaries
    combined_summary = "\n\n".join(chunk_summaries)

    # ---------------- REDUCE STAGE ----------------

    combine_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting summarizer.

            Combine the provided partial summaries into one
            clear, professional and well-structured meeting summary.

            Organize the final summary using these sections:

            1. Meeting Overview
            2. Key Discussion Points
            3. Important Decisions
            4. Action Items
            5. Deadlines and Next Steps

            Instructions:
            - Remove repetitive information.
            - Preserve important facts and details.
            - Maintain the original meaning.
            - Do not invent information.
            - Do not assume missing responsibilities or deadlines.
            - If a section has no available information, mention
              that it was not specified.

            Return the final summary in readable bullet points.
            """
        ),
        ("human", "{text}")
    ])

    combine_chain = combine_prompt | llm | StrOutputParser()

    final_summary = combine_chain.invoke({
        "text": combined_summary
    })

    return final_summary.strip()


# --------------------------------------------------
# 4. Generate Meeting Title
# --------------------------------------------------

def generate_title(summary: str) -> str:
    """Generate a concise and descriptive meeting title."""

    if not summary or not summary.strip():
        raise ValueError("Summary cannot be empty.")

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting summarizer.

            Generate a short, professional and descriptive
            title based on the meeting summary.

            Instructions:
            - Keep the title between 5 and 12 words.
            - Capture the main purpose of the meeting.
            - Return only the title.
            - Do not include quotation marks or explanations.
            """
        ),
        ("human", "{text}")
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    title = title_chain.invoke({
        "text": summary[:2000]
    })

    return title.strip().strip('"').strip("'")
