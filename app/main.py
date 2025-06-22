# app/main.py
# ------------------------------------------------------------------------
# Micro-OCR para formularios impresos de encuestas (canal 4 – papel)
# ------------------------------------------------------------------------
import os, io, uuid, json, base64, logging, requests
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
from openai import OpenAI

import cv2, numpy as np
try:
    from pyzbar.pyzbar import decode as decode_zbar
    _HAS_ZBAR = True
except ImportError:
    _HAS_ZBAR = False

import boto3
from botocore.exceptions import ClientError

# ─────────────────────────── Config ─────────────────────────────────────
load_dotenv()

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
BACKEND_BASE_URL  = os.getenv("BACKEND_BASE_URL")   # p. ej. http://localhost:8000
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION        = os.getenv("AWS_REGION")
S3_BUCKET         = os.getenv("S3_BUCKET")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY faltante")
if not BACKEND_BASE_URL:
    raise RuntimeError("BACKEND_BASE_URL faltante")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS,
    region_name=AWS_REGION,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
app = FastAPI(title="Micro-OCR encuestas (papel)")

# ─────────────────────────── Utilidades ─────────────────────────────────
def upload_to_s3(content: bytes, filename: str, ctype: str) -> str:
    try:
        s3.upload_fileobj(io.BytesIO(content), S3_BUCKET, filename,
                          ExtraArgs={"ContentType": ctype})
        return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{filename}"
    except ClientError as e:
        raise HTTPException(500, f"Error S3: {e}")

def extract_qr(img_bytes: bytes) -> str:
    np_img = np.frombuffer(img_bytes, np.uint8)
    cv_img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    det    = cv2.QRCodeDetector()

    if hasattr(det, "detectAndDecodeMulti"):
        ok, data_list, _, _ = det.detectAndDecodeMulti(cv_img)
        if ok and data_list:
            txt = next((d for d in data_list if d), "")
            if txt:
                return txt.strip()

    txt, _, _ = det.detectAndDecode(cv_img)
    if txt:
        return txt.strip()

    if _HAS_ZBAR:
        for sym in decode_zbar(cv_img):
            if sym.data:
                return sym.data.decode().strip()

    raise HTTPException(400, "No se detectó un QR legible")

def get_plantilla_map(entrega_id: str) -> Dict[str, Any]:
    url  = f"https://{BACKEND_BASE_URL}/public/entregas/{entrega_id}/plantilla-mapa"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, "Backend no devolvió plantilla")
    return resp.json()

# ───────────── Sanitizado de la respuesta Vision ───────────────────────
def sanitize_answers(data: Dict[str, Any], plantilla: Dict[str, Any]) -> Dict[str, Any]:
    mapa = {p["id"]: p for p in plantilla["preguntas"]}
    dep: List[Dict[str, Any]] = []

    for idx, item in enumerate(data.get("respuestas_preguntas", []), 1):
        qid = item.get("pregunta_id")
        if not qid:
            logging.warning("Item %s sin pregunta_id → descartado: %s", idx, item)
            continue

        tipo = item.get("tipo_pregunta_id")
        preg = mapa.get(qid) or {}

        # — tipo 1: quitar duplicados
        if tipo == 1 and item.get("texto"):
            if item["texto"].strip().lower() == preg.get("texto", "").lower():
                item["texto"] = None

        # — tipo 2: solo número
        if tipo == 2 and item.get("numero") is not None:
            item["texto"] = None

        # — tipo 3 (única): dejar solo una opción y sin texto
        if tipo == 3:
            item["texto"] = None
            if item.get("opciones_ids"):
                item["opcion_id"] = item["opciones_ids"][0]
                item["opciones_ids"] = []

        # — tipo 4 (multi): filtrar UUID válidos y sin texto
        if tipo == 4:
            item["texto"] = None
            valid = {o["id"] for o in preg.get("opciones", [])}
            item["opciones_ids"] = [oid for oid in item.get("opciones_ids", []) if oid in valid]

        dep.append(item)

    data["respuestas_preguntas"] = dep
    return data

# ───────── Construir payload para el backend ────────────────────────────
def build_backend_payload(ocr_json: Dict[str, Any]) -> Dict[str, Any]:
    filas: List[Dict[str, Any]] = []

    for r in ocr_json.get("respuestas_preguntas", []):
        qid = r["pregunta_id"]
        base: Dict[str, Any] = {"pregunta_id": qid}

        if r.get("numero") is not None:
            base["numero"] = float(r["numero"])       # ← serializable
        elif r.get("texto"):
            base["texto"]  = r["texto"].strip()

        # multiselect → varias filas
        if r.get("opciones_ids"):
            for oid in r["opciones_ids"]:
                filas.append({**base, "opcion_id": oid})
        else:
            if oid := r.get("opcion_id"):
                base["opcion_id"] = oid
            filas.append(base)

    return {"respuestas_preguntas": filas}

# ───────────────────────── Endpoint /ocr ────────────────────────────────
@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")

    img_bytes = await file.read()

    entrega_id = extract_qr(img_bytes)
    plantilla  = get_plantilla_map(entrega_id)
    preguntas  = plantilla["preguntas"]

    prompt = (
        "Eres un OCR para formularios impresos de encuestas.\n\n"
        "Devuelve SOLO un JSON con la clave «respuestas_preguntas».\n"
        "Cada elemento debe tener: pregunta_id, tipo_pregunta_id, texto, numero, "
        "opcion_id, opciones_ids.\n"
        "• tipo 1 → texto  • tipo 2 → numero\n"
        "• tipo 3 → opción única (opcion_id)  • tipo 4 → varias opciones_ids\n"
        "No repitas el texto de la pregunta.\n\n"
        f"Plantilla:\n{json.dumps(preguntas, ensure_ascii=False)}"
    )

    b64      = base64.b64encode(img_bytes).decode()
    data_url = f"data:{file.content_type};base64,{b64}"

    try:
        comp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Asistente OCR de encuestas en papel."},
                {"role": "user",   "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        ocr_raw = json.loads(comp.choices[0].message.content.strip())
    except Exception as exc:
        raise HTTPException(500, f"Error OpenAI: {exc}")

    ocr_json        = sanitize_answers(ocr_raw, plantilla)
    backend_payload = build_backend_payload(ocr_json)

    # subir imagen
    fname  = f"{uuid.uuid4().hex}.{file.filename.split('.')[-1]}"
    img_url = upload_to_s3(img_bytes, fname, file.content_type)

    # POST backend
    try:
        post_url = f"{BACKEND_BASE_URL}/public/entregas/{entrega_id}/respuestas"
        be_resp  = requests.post(post_url, json=backend_payload, timeout=10)
        be_resp.raise_for_status()
        backend_res = be_resp.json()
    except Exception as exc:
        backend_res = {"error": str(exc), "payload_enviado": backend_payload}

    # respuesta final (jsonable_encoder para UUID/Decimal/etc.)
    out = {
        "entrega_id":        entrega_id,
        "ocr_result":        ocr_json,
        "payload_enviado":   backend_payload,
        "respuesta_backend": backend_res,
        "plantilla_usada":   plantilla,
        "image_url":         img_url,
    }
    return JSONResponse(content=jsonable_encoder(out))

# Health-check
@app.get("/health")
async def health():
    return {"status": "ok"}
