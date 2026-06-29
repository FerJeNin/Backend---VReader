import base64
import io
import logging
import numpy as np
from PIL import Image
import easyocr

logger = logging.getLogger(__name__)

_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Inicializando EasyOCR...")
        _ocr_engine = easyocr.Reader(['es', 'en'], gpu=False)
        logger.info("EasyOCR listo.")
    return _ocr_engine

def _image_bytes_to_array(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

def _image_base64_to_array(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return _image_bytes_to_array(image_bytes)

def _parse_result(result):
    if not result:
        return {"text": "", "blocks": [], "confidence": 0.0}
    blocks, lines, confidences = [], [], []
    for (bbox, text, confidence) in result:
        if confidence < 0.3:
            continue
        blocks.append({"text": text, "confidence": round(float(confidence), 4), "bbox": bbox})
        lines.append(text)
        confidences.append(confidence)
    return {
        "text": "\n".join(lines),
        "blocks": blocks,
        "confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
    }

def extract_text_from_bytes(image_bytes):
    try:
        result = get_ocr_engine().readtext(_image_bytes_to_array(image_bytes))
        return {**_parse_result(result), "success": True, "error": None}
    except Exception as e:
        logger.error(f"Error OCR: {e}")
        return {"text": "", "blocks": [], "confidence": 0.0, "success": False, "error": str(e)}

def extract_text_from_base64(b64_image):
    try:
        result = get_ocr_engine().readtext(_image_base64_to_array(b64_image))
        return {**_parse_result(result), "success": True, "error": None}
    except Exception as e:
        logger.error(f"Error OCR: {e}")
        return {"text": "", "blocks": [], "confidence": 0.0, "success": False, "error": str(e)}
