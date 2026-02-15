import logging
import PyPDF2
import numpy as np
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)
chunks = []
embeddings = []


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks (simple word-based chunking)"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


def embed_chunks(text_chunks, client):
    """Get embeddings from Mistral API"""
    response = client.embeddings(
            model="mistral-embed",
            input=text_chunks
        )
    data = response.data

    return [item.embedding for item in response.data]


def ingest_pdf(pdf_file, client):
    logger.info("Starting ingestion process...")
    """Main ingestion pipeline"""
    global chunks, embeddings
    
    # Extract text
    text = extract_text_from_pdf(pdf_file)
    logger.info("Extracted text length: %s characters", len(text))
    
    # Chunk text
    new_chunks = chunk_text(text)
    logger.info("Created %s chunks", len(new_chunks))
    
    # Get embeddings
    new_embeddings = embed_chunks(new_chunks, client)
    logger.info("Generated embeddings for %s chunks", len(new_embeddings))
    
    # Store
    chunks.extend(new_chunks)
    embeddings.extend(new_embeddings)
    
    return len(new_chunks)