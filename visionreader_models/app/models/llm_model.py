"""
VisionReader — Módulo LLM
Usa OpenRouter para acceder a modelos de lenguaje.
Operaciones: responder preguntas sobre el texto OCR y generar resúmenes.
"""

import logging
from enum import Enum
from openai import OpenAI  # OpenRouter es compatible con el cliente openai

logger = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Modelo por defecto. Puedes cambiar a cualquiera disponible en OpenRouter:
# "google/gemini-flash-1.5", "meta-llama/llama-3.1-8b-instruct:free", etc.
DEFAULT_MODEL = "deepseek/deepseek-r1:free"

# Límites
MAX_TOKENS_SUMMARY = 512
MAX_TOKENS_QA = 1024
MAX_TEXT_LENGTH = 8000  # Caracteres máximos del texto OCR a enviar al LLM


# ── Cliente ────────────────────────────────────────────────────────────────────

def get_llm_client(api_key: str) -> OpenAI:
    """Crea un cliente OpenAI apuntando a OpenRouter."""
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


# ── Prompts del sistema ────────────────────────────────────────────────────────

class SystemPrompt(str, Enum):
    SUMMARIZE = (
        "Eres un asistente de accesibilidad para personas con discapacidad visual. "
        "Tu tarea es generar resúmenes claros, concisos y fáciles de escuchar en voz alta. "
        "Usa oraciones cortas. Evita listas con viñetas. Responde siempre en español "
        "a menos que el texto original esté en otro idioma."
    )
    QA = (
        "Eres un asistente de accesibilidad para personas con discapacidad visual. "
        "El usuario te proporcionará texto extraído de una imagen y te hará preguntas sobre él. "
        "Responde de forma directa y clara, optimizada para ser leída en voz alta. "
        "Si la respuesta no está en el texto, dilo explícitamente. "
        "Responde siempre en español a menos que el usuario pregunte en otro idioma."
    )


# ── Funciones principales ──────────────────────────────────────────────────────

def summarize_text(extracted_text: str, api_key: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Genera un resumen del texto extraído por OCR, optimizado para lectura en voz alta.

    Args:
        extracted_text: Texto plano devuelto por el módulo OCR.
        api_key:        API key de OpenRouter.
        model:          Modelo a usar (por defecto gpt-4o-mini).

    Retorna:
        {
            "summary": str,
            "model_used": str,
            "success": bool,
            "error": str | None,
        }
    """
    if not extracted_text.strip():
        return {
            "summary": "No se detectó texto en la imagen.",
            "model_used": model,
            "success": True,
            "error": None,
        }

    # Truncar texto muy largo para no exceder el contexto
    text_to_send = extracted_text[:MAX_TEXT_LENGTH]
    if len(extracted_text) > MAX_TEXT_LENGTH:
        text_to_send += "\n[Texto truncado por longitud]"

    try:
        client = get_llm_client(api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS_SUMMARY,
            messages=[
                {"role": "system", "content": SystemPrompt.SUMMARIZE.value},
                {
                    "role": "user",
                    "content": (
                        f"Resume el siguiente texto de forma clara y concisa:\n\n"
                        f"{text_to_send}"
                    ),
                },
            ],
        )
        summary = response.choices[0].message.content.strip()
        return {
            "summary": summary,
            "model_used": response.model,
            "success": True,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Error en LLM summarize: {e}")
        return {
            "summary": "",
            "model_used": model,
            "success": False,
            "error": str(e),
        }


def answer_question(
    extracted_text: str,
    question: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Responde una pregunta del usuario basándose en el texto OCR.

    Args:
        extracted_text:       Texto extraído por OCR.
        question:             Pregunta del usuario.
        api_key:              API key de OpenRouter.
        model:                Modelo a usar.
        conversation_history: Lista de turnos previos [{"role": "user"|"assistant", "content": str}]
                              para mantener contexto conversacional.

    Retorna:
        {
            "answer": str,
            "model_used": str,
            "success": bool,
            "error": str | None,
        }
    """
    if not extracted_text.strip():
        return {
            "answer": "No hay texto extraído para responder preguntas.",
            "model_used": model,
            "success": True,
            "error": None,
        }

    text_to_send = extracted_text[:MAX_TEXT_LENGTH]

    # Construir mensajes: sistema + contexto del texto + historial + nueva pregunta
    messages = [
        {"role": "system", "content": SystemPrompt.QA.value},
        {
            "role": "user",
            "content": (
                f"El siguiente texto fue extraído de una imagen mediante OCR:\n\n"
                f"---\n{text_to_send}\n---\n\n"
                f"Usa este texto como única fuente de información para responder mis preguntas."
            ),
        },
        {"role": "assistant", "content": "Entendido. Puedes hacerme preguntas sobre el texto."},
    ]

    # Agregar historial de conversación si existe
    if conversation_history:
        messages.extend(conversation_history)

    # Nueva pregunta del usuario
    messages.append({"role": "user", "content": question})

    try:
        client = get_llm_client(api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS_QA,
            messages=messages,
        )
        answer = response.choices[0].message.content.strip()
        return {
            "answer": answer,
            "model_used": response.model,
            "success": True,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Error en LLM QA: {e}")
        return {
            "answer": "",
            "model_used": model,
            "success": False,
            "error": str(e),
        }
