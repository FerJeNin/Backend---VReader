"""
VisionReader — Módulo OCR
Usa PaddleOCR (PP-OCRv5) para extraer texto de imágenes.
Recibe imagen como bytes (desde cámara o upload) y devuelve texto plano.
"""

import base64
import io
import logging
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

# Instancia global: se carga una sola vez al iniciar el servidor.
# use_angle_cls=True  → corrige texto rotado (útil para fotos de cámara)
# lang="es"           → modelo optimizado para español (también detecta inglés)
# show_log=False      → silencia logs internos de PaddlePaddle
_ocr_engine: PaddleOCR | None = None


def get_ocr_engine() -> PaddleOCR:
    """Devuelve la instancia singleton de PaddleOCR, inicializándola si es necesario."""
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Inicializando PaddleOCR PP-OCRv5...")
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            use_gpu=False,          # Cambiar a True si el servidor tiene GPU
        )
        logger.info("PaddleOCR listo.")
    return _ocr_engine


def _image_bytes_to_array(image_bytes: bytes) -> np.ndarray:
    """Convierte bytes de imagen (JPG/PNG/WEBP) a array NumPy RGB."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def _image_base64_to_array(b64_string: str) -> np.ndarray:
    """Convierte imagen en base64 (con o sin header data:image/...) a array NumPy."""
    # Remover header si viene como data URL: "data:image/jpeg;base64,..."
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return _image_bytes_to_array(image_bytes)


def _parse_ocr_result(result: list) -> dict:
    """
    Parsea la salida cruda de PaddleOCR y devuelve texto limpio + bloques detallados.

    Estructura de salida de PaddleOCR:
    result[0] = lista de detecciones
    Cada detección = [bounding_box, (texto, confianza)]
    """
    if not result or not result[0]:
        return {"text": "", "blocks": [], "confidence": 0.0}

    blocks = []
    lines = []
    confidences = []

    for detection in result[0]:
        bounding_box, (text, confidence) = detection
        if confidence < 0.5:   # Descartar detecciones de baja confianza
            continue
        blocks.append({
            "text": text,
            "confidence": round(float(confidence), 4),
            "bbox": bounding_box,
        })
        lines.append(text)
        confidences.append(confidence)

    full_text = "\n".join(lines)
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    return {
        "text": full_text,
        "blocks": blocks,
        "confidence": round(avg_confidence, 4),
    }


def extract_text_from_bytes(image_bytes: bytes) -> dict:
    """
    Extrae texto de una imagen dada como bytes.

    Retorna:
        {
            "text": str,           # Texto completo extraído
            "blocks": list,        # Bloques individuales con bbox y confianza
            "confidence": float,   # Confianza promedio (0.0 - 1.0)
            "success": bool,
            "error": str | None,
        }
    """
    try:
        engine = get_ocr_engine()
        image_array = _image_bytes_to_array(image_bytes)
        result = engine.ocr(image_array, cls=True)
        parsed = _parse_ocr_result(result)
        return {**parsed, "success": True, "error": None}
    except Exception as e:
        logger.error(f"Error en OCR (bytes): {e}")
        return {"text": "", "blocks": [], "confidence": 0.0, "success": False, "error": str(e)}


def extract_text_from_base64(b64_image: str) -> dict:
    """
    Extrae texto de una imagen dada como string base64.
    Acepta data URLs: "data:image/jpeg;base64,..."

    Retorna el mismo schema que extract_text_from_bytes.
    """
    try:
        engine = get_ocr_engine()
        image_array = _image_base64_to_array(b64_image)
        result = engine.ocr(image_array, cls=True)
        parsed = _parse_ocr_result(result)
        return {**parsed, "success": True, "error": None}
    except Exception as e:
        logger.error(f"Error en OCR (base64): {e}")
        return {"text": "", "blocks": [], "confidence": 0.0, "success": False, "error": str(e)}
