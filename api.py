import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv
import numpy as np
from groq import Groq

load_dotenv()

app = FastAPI(title="RAG API")
model = SentenceTransformer("all-MiniLM-L6-v2")
groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
supabase = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_ANON_KEY")
)

class Documento(BaseModel):
    texto: str

class Pergunta(BaseModel):
    texto: str

def chunk_texto(texto, tamanho=150, overlap=30):
    palavras = texto.split()
    chunks = []
    i = 0
    while i < len(palavras):
        chunk = " ".join(palavras[i:i + tamanho])
        chunks.append(chunk)
        i += tamanho - overlap
    return chunks

def parse_embedding(emb):
    if isinstance(emb, str):
        return [float(x) for x in emb.strip("[]").split(",")]
    return emb

@app.get("/")
def home():
    total = supabase.table("documentos").select("id", count="exact").execute()
    return {"status": "RAG API rodando", "chunks_indexados": total.count}

@app.post("/indexar")
def indexar(doc: Documento):
    chunks = chunk_texto(doc.texto)
    embeddings = model.encode(chunks)
    for chunk, emb in zip(chunks, embeddings):
        supabase.table("documentos").insert({
            "conteudo": chunk,
            "embedding": emb.tolist()
        }).execute()
    return {"mensagem": "Documento indexado com sucesso", "chunks_criados": len(chunks)}

@app.post("/perguntar")
def perguntar(pergunta: Pergunta):
    rows = supabase.table("documentos").select("conteudo, embedding").execute().data
    if not rows:
        return {"erro": "Nenhum documento indexado. Use /indexar primeiro."}
    chunks = [r["conteudo"] for r in rows]
    embeddings = np.array([parse_embedding(r["embedding"]) for r in rows])
    emb = model.encode([pergunta.texto])[0]
    sims = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )
    indices = np.argsort(sims)[::-1][:2]
    contexto = "\n".join([chunks[i] for i in indices])
    resposta = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Responda usando APENAS o contexto fornecido."},
            {"role": "user", "content": f"Contexto:\n{contexto}\n\nPergunta: {pergunta.texto}"}
        ]
    )
    return {
        "pergunta": pergunta.texto,
        "resposta": resposta.choices[0].message.content,
        "chunks_usados": [chunks[i] for i in indices]
    }
