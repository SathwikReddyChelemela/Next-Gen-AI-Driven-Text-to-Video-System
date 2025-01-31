import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the fine-tuned model and tokenizer
def load_finetuned_model(model_path :str):
    """
    Load the fine-tuned model and tokenizer.

    Args:
        model_path (str): Path to the fine-tuned model directory.

    Returns:
        model, tokenizer: The loaded model and tokenizer.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    logger.info("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    logger.info("Fine-tuned model loaded successfully!")
    return model, tokenizer

# Generate detailed prompt using the fine-tuned model
def enhance_prompt_with_model(prompt: str, model, tokenizer, max_length: int = 300):
    """
    Enhance the RAG prompt using the fine-tuned model.

    Args:
        prompt (str): The input prompt.
        model: The fine-tuned model instance.
        tokenizer: The tokenizer for the fine-tuned model.
        max_length (int): Maximum token length for the generated output.

    Returns:
        str: The enhanced detailed prompt.
    """
    logger.info("Generating detailed description with fine-tuned model...")
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate output with the model
    outputs = model.generate(inputs, max_length=max_length, num_beams=5, early_stopping=True)
    detailed_description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Detailed description generated successfully!")
    return detailed_description
