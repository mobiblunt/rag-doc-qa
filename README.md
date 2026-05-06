# RAG Q&A - Retrieval Augmented Generation Document QA System

A web-based Q&A system that uses **Retrieval Augmented Generation (RAG)** to answer questions about PDF documents. Upload PDFs, ingest them into a vector database, and ask intelligent questions powered by Google's Gemini API.

## Features

🚀 **Easy to Use**
- Drag-and-drop PDF upload
- One-click document ingestion
- Real-time streaming responses

🔍 **Smart Retrieval**
- Local embeddings using SentenceTransformers (`all-MiniLM-L6-v2`)
- Vector similarity search with ChromaDB
- Context-aware responses from relevant document chunks

💬 **Powered by Gemini API**
- Free tier compatible (gemini-2.0-flash)
- Streaming responses for better UX
- Source citations with every answer

🎨 **Modern UI**
- Dark-themed web interface
- Status indicators (idle/busy/error)
- Real-time document list
- Responsive design with grid layout

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Vector DB**: ChromaDB (persistent local storage)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM API**: Google Generative AI (Gemini)
- **PDF Processing**: PyPDF

## Prerequisites

- Python 3.12+
- [Google Gemini API key](https://ai.google.dev/) (free tier)
- 2GB+ disk space for embeddings cache and vector database

## Installation

### 1. Clone and Setup

```bash
git clone <repo-url>
cd rag-doc-qa

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or: source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Get your free API key from: https://ai.google.dev/

## Running the Application

```bash
# Start the FastAPI server
uvicorn main:app --reload

# Open in browser
http://localhost:8000
```

The server will:
- Serve the web interface at `/`
- Preload embeddings model on startup (first run ~30 seconds)
- Accept PDF uploads at `/api/upload`
- Provide Q&A endpoint at `/api/ask`

## How It Works

### Document Ingestion Pipeline

```
PDF Upload → PyPDF Extraction → Text Chunking → Embedding → ChromaDB Storage
```

1. **Upload**: PDFs are uploaded to the `/docs` folder
2. **Extract**: Text is extracted page-by-page using PyPDF
3. **Chunk**: Text is split into 500-token chunks with 50-token overlap
4. **Embed**: Each chunk is embedded using local SentenceTransformers model
5. **Store**: Embeddings + metadata stored in ChromaDB for fast retrieval

### Question Answering

```
User Question → Embed Query → Vector Search → Retrieve Context → Generate Response
```

1. **Embed**: Question is embedded using the same model
2. **Search**: ChromaDB finds 5 most similar chunks using cosine similarity
3. **Context**: Retrieved chunks are combined into a single context prompt
4. **Generate**: Gemini API generates answer from context (streaming)
5. **Sources**: Original PDF filenames are returned with the answer

## Project Structure

```
rag-doc-qa/
├── main.py              # FastAPI application & routes
├── ingest.py            # PDF ingestion pipeline
├── embedder.py          # Embedding model management
├── static/              # Web interface
│   └── index.html       # Frontend UI
├── docs/                # Uploaded PDFs (auto-created)
├── chroma_db/           # Vector database (auto-created)
├── .env                 # API keys (create this)
└── pyproject.toml       # Dependencies
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend (index.html) |
| `GET` | `/health` | Health check |
| `GET` | `/api/documents` | List uploaded PDFs |
| `POST` | `/api/upload` | Upload a PDF file |
| `POST` | `/api/ingest` | Ingest uploaded PDFs |
| `POST` | `/api/ask` | Ask a question (streaming) |

### Example Request

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "n_results": 5}'
```

## Free Tier Limitations

⚠️ **Gemini API Free Tier**
- ~15 requests per minute
- ~1 million tokens per day
- If you hit the rate limit, wait ~60 seconds before retrying

**For higher usage**, upgrade to a paid plan: https://ai.google.dev/pricing

## Configuration

### Model Settings

In `embedder.py`:
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dims, ~90MB)
- Can be swapped for other HuggingFace sentence transformers

In `main.py`:
- **Chunk Size**: 500 tokens (` ingest.py`)
- **Chunk Overlap**: 50 tokens
- **Search Results**: 5 chunks (configurable per request)

### Changing the LLM

Replace `gemini-2.0-flash` in `main.py` with:
- `gemini-pro` (older, might have quota)
- Or use a local LLM with Ollama (no API limits)

## Troubleshooting

### "GEMINI_API_KEY not configured"
- Check `.env` file exists with valid key
- Restart the server after adding key

### Rate limit hit (429 error)
- Free tier has strict limits (~15 req/min)
- Wait 60 seconds before retrying
- Consider upgrading or using a local LLM

### "No documents indexed yet"
- Upload PDFs first via the UI
- Click "Ingest Documents" button
- Wait for ingestion to complete

### Model loading slow
- First embedding load takes ~30 seconds
- Subsequent requests are fast (cached model)

## Performance Tips

1. **Use smaller PDFs** (< 50MB total) for faster ingestion
2. **Adjust chunk size** in `ingest.py` if retrieval is poor:
   - Smaller chunks → more specific results
   - Larger chunks → more context
3. **Batch operations**: Ingestion batches embeddings (100 at a time)

## Future Enhancements

- [ ] Support for other document formats (DOCX, TXT, etc.)
- [ ] Custom chunk size/overlap in UI
- [ ] Export chat history
- [ ] Multi-document comparison
- [ ] Local LLM integration (Ollama, LLaMA.cpp)
- [ ] Authentication & multi-user support

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

---

**Built with ❤️ using FastAPI + ChromaDB + Gemini API**
