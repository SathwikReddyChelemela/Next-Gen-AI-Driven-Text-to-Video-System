from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from knowledge_base import initialize_rag, search_knowledge_base
from finetuned_model import load_finetuned_model, enhance_prompt_with_model
from video_generation import generate_video_from_image
import logging
from dotenv import load_dotenv
import requests
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize RAG
try:
    logger.info("Initializing RAG...")
    vector_store = initialize_rag()
    logger.info("RAG initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize RAG: {e}")
    raise RuntimeError(f"Failed to initialize RAG: {e}")

# Load fine-tuned model
try:
    model_path = "/Users/SathwikReddyChelemela/Documents/finalprompt project/TextToVideo-fork/Backend/fine_tuned_model"  # Update with your model path
    logger.info("Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model(model_path)
    logger.info("Fine-tuned model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load fine-tuned model: {e}")
    raise RuntimeError(f"Failed to load fine-tuned model: {e}")

# OpenAI API for Image Generation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("Missing OpenAI API key. Set the OPENAI_API_KEY environment variable.")
    raise ValueError("OpenAI API key is required.")

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}


def generate_image(prompt: str) -> str:
    """
    Generate an image from a prompt using OpenAI's API.
    """
    logger.info(f"Generating image for prompt: {prompt}")
    url = "https://api.openai.com/v1/images/generations"
    payload = {"prompt": prompt, "n": 1, "size": "1024x1024"}

    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response.text}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {response.text}")

    image_url = response.json()["data"][0]["url"]
    logger.info(f"Image generated successfully: {image_url}")
    return image_url


# Request schema
class GenerateVideoRequest(BaseModel):
    user_input: str


@app.post("/generate_video")
async def generate_video_endpoint(request: GenerateVideoRequest):
    """
    Endpoint to process user input, enhance it with RAG, generate an image, and create a video.
    """
    try:
        user_input = request.user_input.strip()
        logger.info(f"Received user input: {user_input}")

        if not user_input:
            raise HTTPException(status_code=400, detail="Input cannot be empty.")

        # Step 1: Enhance the input using RAG
        enhanced_prompt = search_knowledge_base(vector_store, user_input)
        logger.info(f"Enhanced prompt from RAG: {enhanced_prompt}")

        # Step 2: Generate a detailed description using the fine-tuned model
        detailed_description = enhance_prompt_with_model(enhanced_prompt, model, tokenizer)

        # Truncate if detailed description exceeds 512 characters
        if len(detailed_description) > 512:
            logger.warning("Detailed description exceeds 512 characters. Truncating...")
            detailed_description = detailed_description[:512]

        logger.info(f"Generated detailed description: {detailed_description}")

        # Step 3: Generate an image using OpenAI
        image_url = generate_image(detailed_description)

        # Step 4: Generate video using Runway ML
        video_url = generate_video_from_image(image_url, detailed_description)
        logger.info(f"Generated video URL: {video_url}")

        # Return only the video URL
        return {"video_url": video_url}

    except Exception as e:
        logger.error(f"Error processing input: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
