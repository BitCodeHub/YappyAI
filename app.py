import os
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting on port {port}", flush=True)
    print(f"PORT env var: {os.environ.get('PORT', 'not set')}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)