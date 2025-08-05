from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import asyncio
import torch
import time
from datetime import datetime

# Import our modules
from generate import TextGenerator
from config import config

# Initialize FastAPI app
app = FastAPI(
    title="GPT Text Generation API",
    description="REST API for generating text using a trained GPT model",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# Add CORS middleware to allow web browsers to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global text generator instance
text_generator = None

# Request/Response models using Pydantic
class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Text prompt to generate from", min_length=1, max_length=500)
    max_length: Optional[int] = Field(100, description="Maximum tokens to generate", ge=1, le=500)
    temperature: Optional[float] = Field(0.8, description="Sampling temperature", ge=0.1, le=2.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling", ge=0, le=100)
    top_p: Optional[float] = Field(0.9, description="Top-p sampling", ge=0.1, le=1.0)

class GenerationResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str
    prompt: str
    generation_time: float
    timestamp: str
    parameters: dict

class BatchGenerationRequest(BaseModel):
    """Request model for batch text generation"""
    prompts: List[str] = Field(..., description="List of prompts", min_items=1, max_items=10)
    max_length: Optional[int] = Field(100, description="Maximum tokens to generate", ge=1, le=500)
    temperature: Optional[float] = Field(0.8, description="Sampling temperature", ge=0.1, le=2.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling", ge=0, le=100)
    top_p: Optional[float] = Field(0.9, description="Top-p sampling", ge=0.1, le=1.0)

class BatchGenerationResponse(BaseModel):
    """Response model for batch text generation"""
    results: List[GenerationResponse]
    total_time: float
    timestamp: str

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    vocab_size: int
    max_length: int
    parameters: dict
    device: str
    status: str

# Startup event to initialize the model
@app.on_event("startup")
async def startup_event():
    """Initialize the text generator when API starts"""
    global text_generator
    print("Initializing text generator...")
    try:
        text_generator = TextGenerator()
        print("Text generator initialized successfully!")
    except Exception as e:
        print(f"Error initializing text generator: {e}")
        # Continue anyway - error will be handled in endpoints

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running and model is loaded"""
    if text_generator is None:
        raise HTTPException(status_code=503, detail="Text generator not initialized")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": text_generator.model is not None
    }

# Model information endpoint
@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if text_generator is None:
        raise HTTPException(status_code=503, detail="Text generator not initialized")
    
    return ModelInfo(
        model_name="GPT Text Generator",
        vocab_size=text_generator.model.vocab_size,
        max_length=config.max_length,
        parameters={
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "d_model": config.d_model,
            "d_ff": config.d_ff
        },
        device=str(text_generator.device),
        status="ready"
    )

# Single text generation endpoint
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text from a single prompt"""
    if text_generator is None:
        raise HTTPException(status_code=503, detail="Text generator not initialized")
    
    try:
        start_time = time.time()
        
        # Generate text using our TextGenerator
        generated_text = text_generator.generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        
        generation_time = time.time() - start_time
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            generation_time=generation_time,
            timestamp=datetime.now().isoformat(),
            parameters={
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Batch text generation endpoint
@app.post("/generate/batch", response_model=BatchGenerationResponse)
async def generate_batch(request: BatchGenerationRequest):
    """Generate text for multiple prompts"""
    if text_generator is None:
        raise HTTPException(status_code=503, detail="Text generator not initialized")
    
    try:
        start_time = time.time()
        results = []
        
        for prompt in request.prompts:
            prompt_start_time = time.time()
            
            generated_text = text_generator.generate_text(
                prompt=prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p
            )
            
            prompt_generation_time = time.time() - prompt_start_time
            
            results.append(GenerationResponse(
                generated_text=generated_text,
                prompt=prompt,
                generation_time=prompt_generation_time,
                timestamp=datetime.now().isoformat(),
                parameters={
                    "max_length": request.max_length,
                    "temperature": request.temperature,
                    "top_k": request.top_k,
                    "top_p": request.top_p
                }
            ))
        
        total_time = time.time() - start_time
        
        return BatchGenerationResponse(
            results=results,
            total_time=total_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

# Creative prompts endpoint
@app.get("/generate/creative")
async def generate_creative():
    """Generate text for predefined creative prompts"""
    if text_generator is None:
        raise HTTPException(status_code=503, detail="Text generator not initialized")
    
    creative_prompts = [
        "In a world where dreams become reality",
        "The last book in the library contained",
        "When artificial intelligence gained consciousness",
        "The secret message hidden in the constellation"
    ]
    
    try:
        results = []
        start_time = time.time()
        
        for prompt in creative_prompts:
            prompt_start_time = time.time()
            
            generated_text = text_generator.generate_text(
                prompt=prompt,
                max_length=80,
                temperature=0.9,  # More creative
                top_k=40,
                top_p=0.85
            )
            
            prompt_generation_time = time.time() - prompt_start_time
            
            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "generation_time": prompt_generation_time
            })
        
        total_time = time.time() - start_time
        
        return {
            "results": results,
            "total_time": total_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Creative generation failed: {str(e)}")

# Simple GET endpoint for quick testing
@app.get("/generate/simple")
async def generate_simple(prompt: str, max_length: int = 50, temperature: float = 0.8):
    """Simple GET endpoint for quick text generation testing"""
    if text_generator is None:
        raise HTTPException(status_code=503, detail="Text generator not initialized")
    
    if not prompt or len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    try:
        generated_text = text_generator.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9
        )
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Root endpoint with API information
@app.get("/")
async def root():
    """API root endpoint with basic information"""
    return {
        "message": "GPT Text Generation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "generate": "/generate (POST)",
            "batch_generate": "/generate/batch (POST)",
            "creative": "/generate/creative",
            "simple": "/generate/simple?prompt=your_prompt",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

# Run the API server
if __name__ == "__main__":
    print("Starting GPT Text Generation API...")
    print("API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    print("ReDoc documentation at: http://localhost:8000/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Accept connections from any IP
        port=8000,       # Default port
        reload=False,    # Set to True for development
        log_level="info"
    )