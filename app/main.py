import io
import os
import logging

from mistralai.client import MistralClient
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from app.ingestion import ingest_pdf
from app.retrieval import detect_intent, transform_query, hybrid_search
from app.generation import generate_answer

from dotenv import load_dotenv
load_dotenv()  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    logger.warning("MISTRAL_API_KEY not found in environment variables. Please set it in your .env file.")

client = MistralClient(api_key=MISTRAL_API_KEY)

app = FastAPI(title="Custom RAG Pipeline API")


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# For defining structure and validating input 
class QueryRequest(BaseModel):
    query: str


# Typed response keeps the API contract stable and documents output fields
class QueryResponse(BaseModel):
    answer: str
    chunks: List[dict]
    needs_search: bool


@app.post("/ingest")
async def ingest_endpoint(files: List[UploadFile] = File(...)):
    """Upload and ingest PDF files"""
    try:
        total_chunks = 0
        logger.info("Received %s file(s) for ingestion.", len(files))
        
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(400, f"File {file.filename} is not a PDF")
            
            # Read file content
            content = await file.read()
            pdf_file = io.BytesIO(content)
            
            # Ingest
            logger.debug("Received PDF file: %s", file.filename)
            num_chunks = ingest_pdf(pdf_file, client)
            total_chunks += num_chunks
        
        return {
            "message": f"Successfully ingested {len(files)} file(s)",
            "total_chunks": total_chunks
        }
    
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the knowledge base"""
    try:
        query = request.query
        logger.info("Received query: %s", query)
        
        # Intent detection
        needs_search = detect_intent(query)
        
        if not needs_search:
            return QueryResponse(
                answer="Hello! Please ask a question about the uploaded documents.",
                chunks=[],
                needs_search=False
            )
        
        # Transform query
        processed_query = transform_query(query)
        
        # Hybrid search
        retrieved_chunks = hybrid_search(processed_query, client)
        
        # Generate answer
        result = generate_answer(query, client, retrieved_chunks)
        
        return QueryResponse(
            answer=result["answer"],
            chunks=result["chunks"],
            needs_search=True
        )
    
    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "healthy"}