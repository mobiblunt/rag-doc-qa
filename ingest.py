import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from embedder import embed_batch


def load_pdfs(folder="docs"):
    documents = []
    if not os.path.exists(folder):
        os.makedirs(folder)
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            reader = PdfReader(f"{folder}/{filename}")
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            documents.append({"filename": filename, "text": text})
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "id": f"{doc['filename']}_chunk_{i}",
                "text": split,
                "source": doc["filename"]
            })
    return chunks


def ingest_pdfs(folder="docs"):
    docs = load_pdfs(folder)
    if not docs:
        return {"message": "No PDF files found", "chunks": 0, "files": []}

    chunks = chunk_documents(docs)

    # Batch embed — much faster than embedding one by one
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_batch(texts)

    # Store in ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("documents")
    except Exception:
        pass

    collection = client.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}  # cosine similarity for normalized embeddings
    )

    # Add in batches of 100 to avoid memory spikes
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c["text"] for c in batch_chunks],
            metadatas=[{"source": c["source"]} for c in batch_chunks]
        )

    return {
        "message": "Ingestion complete",
        "chunks": len(chunks),
        "files": [d["filename"] for d in docs]
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    result = ingest_pdfs()
    print(result)