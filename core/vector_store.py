import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcripts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    """Load the local Hugging Face embedding model."""

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)




def build_vector_store(transcript: str) -> Chroma:
    """Split the transcript, embed the chunks and store them in ChromaDB."""

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    print("Building vector store...")

    # 1. Split transcript into small overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(transcript)

    # 2. Wrap each chunk in a Document with metadata
    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    # 3. Remove the previous meeting's data so transcripts don't mix
    old_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    old_store.delete_collection()

    # 4. Create a fresh collection and save it to disk
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    print(f"Vector store ready with {len(docs)} chunks.")
    return vector_store




def load_vector_store() -> Chroma:
    """Reopen a previously saved vector store."""

    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vector store directory '{CHROMA_DIR}' does not exist. "
            "Please build the vector store first."
        )

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME
    )




def get_retriever(vector_store: Chroma, k: int = 4):
    """Return a retriever that fetches the k most similar chunks."""

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )