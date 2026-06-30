from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

# 1. Cargar la API Key oculta desde el archivo .env
load_dotenv()
API_KEY = os.getenv("OCR_API_KEY")

app = FastAPI(title="Backend Lector Accesible")

# 2. Configurar CORS (Crucial para que tu HTML local pueda hablar con Python)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, aquí pondrías el dominio real de tu web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Definir el formato de los datos que recibiremos
class ImagenRequest(BaseModel):
    base64Image: str

# 4. El Endpoint principal
@app.post("/api/extraer-texto")
async def extraer_texto(data: ImagenRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Falta configurar la API Key en el servidor.")

    # Preparamos el paquete para enviarlo a OCR.Space
    payload = {
        'base64Image': data.base64Image,
        'language': 'spa',
        'apikey': API_KEY,
        'isOverlayRequired': 'false'
    }

    try:
        # Tu servidor (no el navegador del usuario) hace la petición a la nube
        respuesta = requests.post('https://api.ocr.space/parse/image', data=payload)
        datos = respuesta.json()

        # Revisar si OCR.Space devolvió algún error interno
        if datos.get('IsErroredOnProcessing'):
            error_msg = datos.get('ErrorMessage', ['Error desconocido en OCR'])[0]
            raise HTTPException(status_code=400, detail=error_msg)

        # Extraer y devolver solo el texto limpio
        resultados = datos.get('ParsedResults', [])
        if resultados:
            texto_limpio = resultados[0].get('ParsedText', '').strip()
            return {"texto": texto_limpio}
        else:
            return {"texto": ""}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de conexión: {str(e)}")
