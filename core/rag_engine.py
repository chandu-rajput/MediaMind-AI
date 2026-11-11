
import os
from dotenv import  load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm import get_llm

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)

load_dotenv()


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


def format_docs(docs):
    """Combine retrieved documents into a single text."""

    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever):
    """Create the RAG pipeline using a retriever and Mistral LLM."""

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
        ),
        ("human", "{question}")
    ])

    # Build the RAG pipeline using LCEL
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def build_rag_chain(transcript: str):
    """Build a new vector store and create the RAG chain."""

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k=4)

    return create_rag_chain(retriever)


def load_rag_chain():
    """Load an existing vector store and create the RAG chain."""

    vector_store = load_vector_store()

    retriever = get_retriever(vector_store, k=4)

    return create_rag_chain(retriever)


def ask_question(rag_chain, question: str) -> str:
    """Execute the RAG chain and return the answer."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    print(f"Question: {question}")

    answer = rag_chain.invoke(question)

    print(f"Answer: {answer}")

    return answer
