import uvicorn
from app import app  # Import your FastAPI app from app.py
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("Server stopped.")
        pass
