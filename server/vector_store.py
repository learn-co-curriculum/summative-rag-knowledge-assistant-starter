from typing import Any, List, Optional, Dict

import chromadb
import requests
import hashlib
import random

from config import Config
from documents import DocumentChunk


def get_chroma_client():
    """
    Create and return a persistent Chroma client.
    """
    return chromadb.PersistentClient(path=Config.CHROMA_PATH)


def get_or_create_collection():
    """
    Get or create the Chroma collection for the knowledge assistant.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name=Config.COLLECTION_NAME)


def get_deterministic_mock_embedding(text: str, dimension: int = 768) -> List[float]:
    """
    Generate a deterministic mock embedding vector of the given dimension.
    """
    hasher = hashlib.sha256(text.encode("utf-8"))
    seed = int(hasher.hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dimension)]


def get_embedding(text: str) -> List[float]:
    """
    Create an embedding for a piece of text using the local model service.

    Attempts the modern Ollama `/api/embed` endpoint with fallback to the legacy `/api/embeddings` endpoint.
    If Ollama is not running, falls back to a deterministic mock embedding.
    """
    # 1. Try modern Ollama /api/embed (Ollama 0.2+) with a short timeout to fail fast if offline
    try:
        url = f"{Config.OLLAMA_BASE_URL.rstrip('/')}/api/embed"
        payload = {
            "model": Config.EMBEDDING_MODEL,
            "input": text
        }
        res = requests.post(url, json=payload, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if "embeddings" in data:
                embeds = data["embeddings"]
                if isinstance(embeds, list) and len(embeds) > 0:
                    if isinstance(embeds[0], list):
                        return embeds[0]
                    return embeds
            if "embedding" in data:
                return data["embedding"]
    except Exception:
        pass

    # 2. Fallback to legacy `/api/embeddings` with a short timeout
    try:
        url = f"{Config.OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
        payload = {
            "model": Config.EMBEDDING_MODEL,
            "prompt": text
        }
        res = requests.post(url, json=payload, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if "embedding" in data:
                return data["embedding"]
            elif "embeddings" in data:
                return data["embeddings"]
    except Exception:
        pass

    # 3. Fallback to deterministic mock embedding if Ollama is unavailable
    return get_deterministic_mock_embedding(text)


def seed_vector_store(chunks: List[DocumentChunk]) -> int:
    """
    Add document chunks to the Chroma collection.
    """
    collection = get_or_create_collection()
    
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    
    for chunk in chunks:
        ids.append(chunk.id)
        documents.append(chunk.text)
        metadatas.append({
            "source": chunk.source,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index
        })
        embeddings.append(get_embedding(chunk.text))
        
    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
    return len(ids)


def retrieve_relevant_chunks(question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks for a user question.
    """
    if top_k is None:
        top_k = Config.TOP_K
        
    collection = get_or_create_collection()
    query_emb = get_embedding(question)
    
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    
    formatted_chunks = []
    if not results or not results.get("ids") or len(results["ids"][0]) == 0:
        return []
        
    ids = results["ids"][0]
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else []
    
    for i in range(len(ids)):
        meta = metadatas[i] or {}
        dist = distances[i] if i < len(distances) else None
        formatted_chunks.append({
            "text": docs[i],
            "source": meta.get("source", "unknown"),
            "title": meta.get("title", "Unknown Source"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist
        })
        
    return formatted_chunks
