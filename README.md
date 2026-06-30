# RAG API 🔍

REST API that answers questions about any document using Retrieval-Augmented Generation (RAG).

## How it works

1. Send a document to `/indexar` — it gets chunked and embedded into pgvector (Supabase)
2. Ask a question to `/perguntar` — the API finds the most relevant chunks and sends them to an LLM
3. Get a precise answer grounded in your document

## Tech Stack

- **API:** FastAPI
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database:** pgvector via Supabase
- **LLM:** Groq (llama-3.3-70b-versatile)

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status and chunk count |
| POST | `/indexar` | Index a document |
| POST | `/perguntar` | Ask a question |

## Run locally

```bash
git clone https://github.com/faustinoluca-spec/rag-api
cd rag-api
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
uvicorn api:app --reload
```

## Environment Variables

```env
GROQ_API_KEY=
SUPABASE_URL=
SUPABASE_ANON_KEY=
```

## Author

Luca Faustino — AI Engineer  
[github.com/faustinoluca-spec](https://github.com/faustinoluca-spec)
