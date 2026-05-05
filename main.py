import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
from embedder import embed, get_model

load_dotenv()

app = FastAPI(title="RAG Q&A")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)


@app.on_event("startup")
def preload_model():
    """Load the embedding model at startup so the first request isn't slow."""
    get_model()


def get_genai_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def retrieve_chunks(question: str, n_results: int = 5):
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("documents")
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Upload and ingest PDFs first."
        )

    # Embed the query locally — same model used during ingestion
    query_embedding = embed(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents():
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    return {"files": files, "count": len(files)}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"Uploaded {file.filename}", "filename": file.filename}


@app.post("/api/ingest")
def run_ingest():
    from ingest import ingest_pdfs
    try:
        result = ingest_pdfs(folder=UPLOAD_DIR)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QuestionRequest(BaseModel):
    question: str
    n_results: int = 5


@app.post("/api/ask")
def ask_question(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    model = get_genai_model()
    chunks = retrieve_chunks(req.question, req.n_results)

    context = "\n\n---\n\n".join([
        f"Source: {c['source']}\n{c['text']}" for c in chunks
    ])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the documents provided."
Be concise and clear.

CONTEXT:
{context}

QUESTION: {req.question}

ANSWER:"""

    sources = list(set(c["source"] for c in chunks))

    def stream():
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    # Return sources as first line encoded in a special way the frontend can parse
    def stream_with_sources():
        yield f"__SOURCES__{','.join(sources)}__SOURCES__\n"
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    return StreamingResponse(stream_with_sources(), media_type="text/plain")