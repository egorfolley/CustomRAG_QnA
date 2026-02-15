import logging
from app.config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def generate_answer(query, client, retrieved_chunks):
    """Generate answer using Mistral LLM"""
    
    # Debug: print scores
    logger.info("Query: %s", query)
    if retrieved_chunks:
        for i, chunk in enumerate(retrieved_chunks[:3]):
            logger.debug("Chunk %s score: %.4f", i + 1, chunk["score"])
    else:
        logger.warning("No chunks retrieved")
    
    # Check similarity threshold
    if not retrieved_chunks or retrieved_chunks[0]["score"] < SIMILARITY_THRESHOLD:
        logger.warning(
            "Top score %.2f below threshold %.2f",
            retrieved_chunks[0]["score"] if retrieved_chunks else 0,
            SIMILARITY_THRESHOLD,
        )
        return {
            "answer": "Insufficient evidence in the knowledge base to answer this question confidently.",
            "chunks": retrieved_chunks[:3] if retrieved_chunks else [],  # Return chunks anyway for debugging
            "reasoning": f"Top chunk similarity ({retrieved_chunks[0]['score'] if retrieved_chunks else 0:.2f}) is below threshold ({SIMILARITY_THRESHOLD})"
        }
    
    # Build context from top chunks
    context = "\n\n".join([
        f"[Chunk {i+1}]: {chunk['chunk']}"
        for i, chunk in enumerate(retrieved_chunks[:3])
    ])
    
    # Build prompt
    prompt = f"""Based on the following document excerpts, answer the user's question.

        Context:
        {context}

        Question: {query}

        Instructions:
        - Provide a clear, concise answer based ONLY on the context provided
        - Cite which chunk(s) support your answer (e.g., "According to Chunk 1...")
        - If the context doesn't contain enough information, say so
        - Do not make up information not present in the context

        Answer:
    """
    
    # Call Mistral API
    response = client.chat(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    
    return {
        "answer": answer,
        "chunks": retrieved_chunks[:3],
        "threshold_passed": True
    }