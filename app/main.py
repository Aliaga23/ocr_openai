import os
import base64
import uuid
import json
import io
import boto3
from botocore.exceptions import ClientError

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Necesitas definir OPENAI_API_KEY en .env")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")

client = OpenAI(api_key=api_key)

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

app = FastAPI(title="OCR Service with OpenAI Vision and S3 Storage")

def upload_to_s3(file_content, file_name, content_type):
    try:
        s3_client.upload_fileobj(
            io.BytesIO(file_content),
            S3_BUCKET,
            file_name,
            ExtraArgs={'ContentType': content_type}
        )
        s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{file_name}"
        return s3_url
    except ClientError as e:
        return None

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    data = await file.read()
    b64 = base64.b64encode(data).decode()
    data_url = f"data:{file.content_type};base64,{b64}"
    
    file_ext = file.filename.split('.')[-1]
    file_name = f"{uuid.uuid4().hex}.{file_ext}"
    s3_url = upload_to_s3(data, file_name, file.content_type)
    
    if not s3_url:
        raise HTTPException(status_code=500, detail="Error al subir a S3")

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente especializado en OCR para formularios. Extraes preguntas y respuestas de imágenes de formularios escaneados o fotografiados."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analiza esta imagen de formulario y extrae las preguntas y respuestas. Devuelve el resultado en formato JSON con la estructura: {\"respuestas_preguntas\": [{\"pregunta_id\": \"id\", \"tipo_pregunta_id\": número, \"texto\": \"texto\", \"numero\": valor, \"opcion_id\": \"id\"}]}"},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        parsed_response = json.loads(resp.choices[0].message.content.strip())
        
        if "respuestas_preguntas" not in parsed_response:
            parsed_response = {"respuestas_preguntas": []}
            raw_text = resp.choices[0].message.content.strip()
            parsed_response["raw_text"] = raw_text
            
    except json.JSONDecodeError:
        text_content = resp.choices[0].message.content.strip()
        parsed_response = {
            "respuestas_preguntas": [],
            "raw_text": text_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al llamar a OpenAI: {e}")

    return JSONResponse({
        "ocr_result": parsed_response,
        "s3_url": s3_url,
        "model_used": "gpt-4o-mini"
    })

@app.get("/health")
async def health_check():
    return {"status": "healthy"}