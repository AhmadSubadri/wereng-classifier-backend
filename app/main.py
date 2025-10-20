"""
Wereng Classification API
FastAPI backend untuk klasifikasi hama wereng pada tanaman padi
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime

from app.routes import classify, info

# Initialize FastAPI app
app = FastAPI(
    title="Wereng Classification API",
    description="API untuk mengklasifikasi hama wereng pada tanaman padi menggunakan Computer Vision",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(classify.router, prefix="/api", tags=["Classification"])
app.include_router(info.router, prefix="/api", tags=["Model Info"])


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - Health check
    """
    return {
        "status": "active",
        "message": "Wereng Classification API is running",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs"
    }


@app.get("/health", tags=["Root"])
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Wereng Classification API",
        "timestamp": datetime.now().isoformat()
    }


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )