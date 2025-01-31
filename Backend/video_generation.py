from runwayml import RunwayML
from fastapi import HTTPException
import logging
import os
from dotenv import load_dotenv
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")

if not RUNWAY_API_KEY:
    logger.error("Runway ML API key is missing. Ensure RUNWAY_API_KEY is set in the environment variables.")
    raise ValueError("Runway ML API key is required.")

# Initialize RunwayML client
try:
    client = RunwayML(api_key=RUNWAY_API_KEY)
    logger.info("Runway ML client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Runway ML client: {e}")
    raise RuntimeError(f"Failed to initialize Runway ML client: {e}")


def generate_video_from_image(image_url: str, prompt_text: str) -> str:
    """
    Generate a video from an image using the Runway ML API.

    Args:
        image_url (str): The URL of the input image.
        prompt_text (str): The prompt to describe the video generation.

    Returns:
        str: The URL of the generated video.
    """
    try:
        logger.info(f"Creating video generation task for image: {image_url} with prompt: {prompt_text}")

        # Truncate the prompt if it exceeds 512 characters
        if len(prompt_text) > 512:
            logger.warning("Prompt text exceeds 512 characters. Truncating...")
            prompt_text = prompt_text[:512]

        # Create a task
        task = client.image_to_video.create(
            model="gen3a_turbo",  # Replace with the correct model name
            prompt_image=image_url,
            prompt_text=prompt_text,
        )
        logger.info(f"Task created successfully. Task ID: {task.id}")

        # Poll for task completion
        return poll_task_result(task.id)
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise HTTPException(status_code=500, detail=f"Runway ML video generation failed: {str(e)}")


def poll_task_result(task_id: str) -> str:
    """
    Poll the Runway ML task until it is completed and retrieve the video URL.

    Args:
        task_id (str): The ID of the task to poll.

    Returns:
        str: The URL of the generated video.
    """
    try:
        logger.info(f"Polling task with ID: {task_id}")
        while True:
            task = client.tasks.retrieve(id=task_id)
            status = task.status
            logger.info(f"Task status: {status}")

            if status == "SUCCEEDED":
                # Expecting task.output to be a list of URLs
                logger.info(f"Task output: {task.output}")
                if isinstance(task.output, list) and len(task.output) > 0:
                    video_url = task.output[0]  # Assume the first URL is the video URL
                    logger.info(f"Video generation completed successfully: {video_url}")
                    return video_url
                else:
                    raise HTTPException(status_code=500, detail="Runway ML did not return a valid video URL.")
            elif status in ["FAILED", "CANCELED"]:
                raise HTTPException(status_code=500, detail=f"Runway ML task failed with status: {status}")

            time.sleep(5)  # Wait for 5 seconds before polling again
    except Exception as e:
        logger.error(f"Error polling task result: {e}")
        raise HTTPException(status_code=500, detail=f"Runway ML task polling failed: {str(e)}")
