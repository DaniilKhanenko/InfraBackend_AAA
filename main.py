import torch, time, psutil
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel

app = FastAPI()
model_name = "sergeyzh/rubert-mini-frida"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

@app.post("/embed")
def get_embed(data: dict):
    start = time.perf_counter()
    inputs = tokenizer(data["text"], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[:, 0, :].flatten().tolist()
    return {"embedding": out, "inference_time_ms": (time.perf_counter() - start) * 1000}

@app.get("/health")
def health(): 
    return {"status": "ok"}

@app.get("/sys_metrics")
def metrics():
    return {"ram_mb": psutil.Process().memory_info().rss / 1024 / 1024}