"""
VisionReader — Módulo OCR
Usa Google Cloud Vision API para extraer texto de imágenes.
Sin modelos locales, sin consumo de RAM extra.
"""

import base64
import io
import logging
import os
import requests
from PIL import Image

logger = logging.getLogger(__name__)

GOOGLE_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"


def _image_bytes_to_base64(image_bytes: bytes) -> str:
    """Convierte bytes de imagen a base64 puro."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _image_base64_clean(b64_string: str) -> str:
    """Remueve el header data URL si existe."""
    if "," in b64_string:
        return b64_string.split(",", 1)[1]
    return b64_string


def _call_vision_api(b64_image: str, api_key: str) -> dict:
    """Llama a Google Vision API y retorna el resultado crudo."""
    payload = {
        "requests": [
            {
                "image": {"content": b64_image},
                "features": [{"type": "TEXT_DETECTION", "maxResults": 1}],
            }
        ]
    }
    response = requests.post(
        f"{GOOGLE_VISION_URL}?key={api_key}",
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def _parse_vision_result(result: dict) -> dict:
    """Parsea la respuesta de Google Vision y extrae texto y bloques."""
    responses = result.get("responses", [])
    if not responses:
        return {"text": "", "blocks": [], "confidence": 0.0}

    response = responses[0]

    if "error" in response:
        raise ValueError(response["error"].get("message", "Error de Vision API"))

    # fullTextAnnotation tiene el texto completo y bloques detallados
    full_annotation = response.get("fullTextAnnotation", {})
    full_text = full_annotation.get("text", "").strip()

    # textAnnotations[0] tiene el texto completo, los siguientes son palabras
    text_annotations = response.get("textAnnotations", [])
    blocks = []
    confidences = []

    for annotation in text_annotations[1:]:  # Saltar el primero (es el texto completo)
        text = annotation.get("description", "")
        vertices = annotation.get("boundingPoly", {}).get("vertices", [])
        blocks.append({
            "text": text,
            "confidence": 0.99,  # Vision API no retorna confianza por bloque
            "bbox": vertices,
        })
        confidences.append(0.99)

    return {
        "text": full_text,
        "blocks": blocks,
        "confidence": 0.99 if full_text else 0.0,
    }


def extract_text_from_bytes(image_bytes: bytes, api_key: str | None = None) -> dict:
    """
    Extrae texto de imagen dada como bytes usando Google Vision API.
    Si no se pasa api_key, la toma de la variable de entorno GOOGLE_VISION_API_KEY.
    """
    try:
        key = api_key or os.getenv("GOOGLE_VISION_API_KEY")
        if not key:
            raise ValueError("Se requiere GOOGLE_VISION_API_KEY")
        b64 = _image_bytes_to_base64(image_bytes)
        result = _call_vision_api(b64, key)
        parsed = _parse_vision_result(result)
        return {**parsed, "success": True, "error": None}
    except Exception as e:
        logger.error(f"Error OCR: {e}")
        return {"text": "", "blocks": [], "confidence": 0.0, "success": False, "error": str(e)}


def extract_text_from_base64(b64_image: str, api_key: str | None = None) -> dict:
    """
    Extrae texto de imagen dada como base64 usando Google Vision API.
    Acepta data URLs: 'data:image/jpeg;base64,...'
    """
    try:
        key = api_key or os.getenv("GOOGLE_VISION_API_KEY")
        if not key:
            raise ValueError("Se requiere GOOGLE_VISION_API_KEY")
        clean_b64 = _image_base64_clean(b64_image)
        result = _call_vision_api(clean_b64, key)
        parsed = _parse_vision_result(result)
        return {**parsed, "success": True, "error": None}
    except Exception as e:
        logger.error(f"Error OCR: {e}")
        return {"text": "", "blocks": [], "confidence": 0.0, "success": False, "error": str(e)}
