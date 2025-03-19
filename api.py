from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from inference5 import generate_text
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define request model
class PromptRequest(BaseModel):
    input: str

# Sample function to generate a response
def generate_response(prompt: str) -> str:
    return generate_text(prompt)

@app.post("/generate")
def generate(prompt_request: PromptRequest):
    response = generate_response(prompt_request.input)
    return response

@app.options("/generate")
async def preflight_handler():
    """Handle preflight requests for CORS"""
    return {}
