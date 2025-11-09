from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

try:
    MODEL_ID = "juanistos/tinyllama"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    tokenizer = None
    model = None

@app.get("/")
def root():
    return {"message": "Bot físico cargado desde Hugging Face"}

@app.post("/chat")
async def chat(request: Request):
    if tokenizer is None or model is None:
        return JSONResponse(content={"error": "Modelo no disponible"}, status_code=500)

    data = await request.json()
    prompt = data.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JSONResponse(content={"response": response})