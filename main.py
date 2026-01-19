import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_rag import router as rag_router

app = FastAPI(
    title="TILA RAG Backend",
    description="Arabic Grammar RAG API built with Qwen + ChromaDB",
    version="1.0.0"
)

# إعداد CORS للسماح للـ Frontend بالاتصال
# يمكنك تحديد origins معينة في الإنتاج
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تسجيل الراوتر
app.include_router(rag_router)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}
