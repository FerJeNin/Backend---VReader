"""
VisionReader — FastAPI Backend
Endpoints para OCR (PaddleOCR) y LLM (OpenRouter).
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.models.ocr_model import extract_text_from_base64, get_ocr_engine
from app.models.llm_model import summarize_text, answer_question, DEFAULT_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Precarga del modelo OCR al arrancar el servidor ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Precargando PaddleOCR...")
    get_ocr_engine()  # Carga el modelo una sola vez
    yield
    logger.info("Servidor apagado.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VisionReader API",
    description="OCR + LLM para personas con discapacidad visual",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # En producción: reemplaza con tu dominio de Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas Pydantic ──────────────────────────────────────────────────────────

class OCRRequest(BaseModel):
    image_base64: str = Field(..., description="Imagen en base64 (con o sin data URL header)")


class OCRResponse(BaseModel):
    text: str
    confidence: float
    success: bool
    error: str | None = None


class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Texto extraído por OCR")
    model: str = Field(DEFAULT_MODEL, description="Modelo de OpenRouter a usar")


class SummarizeResponse(BaseModel):
    summary: str
    model_used: str
    success: bool
    error: str | None = None


class QuestionRequest(BaseModel):
    text: str = Field(..., description="Texto extraído por OCR")
    question: str = Field(..., description="Pregunta del usuario")
    model: str = Field(DEFAULT_MODEL, description="Modelo de OpenRouter a usar")
    conversation_history: list[dict] | None = Field(
        None, description="Historial de turnos previos para contexto"
    )


class QuestionResponse(BaseModel):
    answer: str
    model_used: str
    success: bool
    error: str | None = None


class OCRAndSummarizeRequest(BaseModel):
    """Endpoint combinado: extrae texto Y genera resumen en una sola llamada."""
    image_base64: str
    model: str = Field(DEFAULT_MODEL)


class OCRAndSummarizeResponse(BaseModel):
    text: str
    confidence: float
    summary: str
    model_used: str
    success: bool
    error: str | None = None


# ── Helper para extraer API key ───────────────────────────────────────────────

def _get_api_key(authorization: str | None) -> str:
    """
    Extrae la API key de OpenRouter del header Authorization.
    Acepta formato: "Bearer sk-or-..." o directamente "sk-or-..."
    Como fallback usa la variable de entorno OPENROUTER_API_KEY.
    """
    if authorization:
        return authorization.replace("Bearer ", "").strip()
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    raise HTTPException(
        status_code=401,
        detail="Se requiere API key de OpenRouter. Envíala en el header Authorization: Bearer <key>",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "VisionReader API"}


@app.post("/ocr", response_model=OCRResponse)
def run_ocr(request: OCRRequest):
    """
    Extrae texto de una imagen en base64 usando PaddleOCR PP-OCRv5.
    No requiere API key.
    """
    result = extract_text_from_base64(request.image_base64)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return OCRResponse(
        text=result["text"],
        confidence=result["confidence"],
        success=True,
    )


@app.post("/summarize", response_model=SummarizeResponse)
def run_summarize(
    request: SummarizeRequest,
    authorization: str | None = Header(None),
):
    """
    Genera un resumen del texto usando el LLM de OpenRouter.
    Requiere API key en el header: Authorization: Bearer <key>
    """
    api_key = _get_api_key(authorization)
    result = summarize_text(request.text, api_key=api_key, model=request.model)
    if not result["success"]:
        raise HTTPException(status_code=502, detail=result["error"])
    return SummarizeResponse(
        summary=result["summary"],
        model_used=result["model_used"],
        success=True,
    )


@app.post("/question", response_model=QuestionResponse)
def run_question(
    request: QuestionRequest,
    authorization: str | None = Header(None),
):
    """
    Responde una pregunta sobre el texto OCR usando el LLM.
    Requiere API key en el header: Authorization: Bearer <key>
    """
    api_key = _get_api_key(authorization)
    result = answer_question(
        extracted_text=request.text,
        question=request.question,
        api_key=api_key,
        model=request.model,
        conversation_history=request.conversation_history,
    )
    if not result["success"]:
        raise HTTPException(status_code=502, detail=result["error"])
    return QuestionResponse(
        answer=result["answer"],
        model_used=result["model_used"],
        success=True,
    )


@app.post("/ocr-and-summarize", response_model=OCRAndSummarizeResponse)
def run_ocr_and_summarize(
    request: OCRAndSummarizeRequest,
    authorization: str | None = Header(None),
):
    """
    Endpoint combinado: extrae texto con OCR y genera resumen en una sola llamada.
    Útil para el flujo principal de VisionReader (captura → leer en voz alta).
    """
    api_key = _get_api_key(authorization)

    ocr_result = extract_text_from_base64(request.image_base64)
    if not ocr_result["success"]:
        raise HTTPException(status_code=500, detail=f"Error OCR: {ocr_result['error']}")

    llm_result = summarize_text(ocr_result["text"], api_key=api_key, model=request.model)

    return OCRAndSummarizeResponse(
        text=ocr_result["text"],
        confidence=ocr_result["confidence"],
        summary=llm_result["summary"],
        model_used=llm_result.get("model_used", request.model),
        success=True,
    )
