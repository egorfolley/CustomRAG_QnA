import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from app.config import TOP_K
from app.ingestion import chunks, embeddings


def detect_intent(query):
    """Simple intent detection - check if query needs search"""
    greetings = ["hello", "hi", "hey", "thanks", "thank you", "bye"]
    query_lower = query.lower().strip()
    
    # Don't search for greetings
    if any(greeting in query_lower for greeting in greetings):
        return False
    
    # Search for actual questions
    return len(query_lower.split()) > 2


def transform_query(query):
    """Simple query transformation"""
    # Basic cleaning
    query = query.strip().lower()
    
    # Expand common abbreviations (optional)
    expansions = {
        "q&a": "question and answer",
        "info": "information",
    }
    
    for abbr, full in expansions.items():
        query = query.replace(abbr, full)
    
    return query


def semantic_search(query, client, top_k=TOP_K):
    """Semantic search using embeddings"""
    if not embeddings:
        return []
    
    # Get query embedding
    query_embedding = client.embeddings(
        model="mistral-embed",
        input=[query]
    ).data[0].embedding
    
    # Calculate cosine similarity
    query_emb_array = np.array(query_embedding).reshape(1, -1)
    chunk_emb_array = np.array(embeddings)
    
    similarities = cosine_similarity(query_emb_array, chunk_emb_array)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = [
        {
            "chunk": chunks[i],
            "score": float(similarities[i]),
            "index": int(i)
        }
        for i in top_indices
    ]
    
    return results


def keyword_search(query, top_k=TOP_K):
    """Keyword search using TF-IDF"""
    if not chunks:
        return []
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks + [query])
    
    # Get query vector (last one)
    query_vector = tfidf_matrix[-1:]
    chunk_vectors = tfidf_matrix[:-1]
    
    # Calculate similarity
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = [
        {
            "chunk": chunks[i],
            "score": float(similarities[i]),
            "index": int(i)
        }
        for i in top_indices
    ]
    
    return results


def hybrid_search(query, client, semantic_weight=0.7, keyword_weight=0.3, top_k=TOP_K):
    """Combine semantic and keyword search"""
    semantic_results = semantic_search(query, client, top_k=top_k * 2)
    keyword_results = keyword_search(query, top_k=top_k * 2)
    
    # Combine scores
    combined_scores = {}
    
    for result in semantic_results:
        idx = result["index"]
        combined_scores[idx] = semantic_weight * result["score"]
    
    for result in keyword_results:
        idx = result["index"]
        if idx in combined_scores:
            combined_scores[idx] += keyword_weight * result["score"]
        else:
            combined_scores[idx] = keyword_weight * result["score"]
    
    # Sort by combined score
    sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top k unique chunks
    final_results = [
        {
            "chunk": chunks[idx],
            "score": score,
            "index": idx
        }
        for idx, score in sorted_indices[:top_k]
    ]
    
    return final_results