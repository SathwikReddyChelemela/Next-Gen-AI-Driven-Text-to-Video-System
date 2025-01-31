from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


load_dotenv()

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is accessible
if not openai_api_key or openai_api_key.strip() == "":
    logger.error("OpenAI API key not found. Ensure OPENAI_API_KEY is set in the environment variables.")
    raise ValueError("Missing OpenAI API key. Set the OPENAI_API_KEY environment variable.")

logger.info("OpenAI API key successfully loaded.")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load and process knowledge base documents
def create_knowledge_base(pdf_file: str):
    logger.info("Loading PDF file...")
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    logger.info("Creating vector store...")
    vector_store = Chroma.from_documents(documents, embeddings)
    logger.info("Knowledge base created successfully!")
    return vector_store

# Search knowledge base
def search_knowledge_base(vector_store, query: str, top_k: int = 1):
    logger.info(f"Searching knowledge base for query: {query}")
    results = vector_store.similarity_search(query, k=top_k)

    # Combine results into an enhanced prompt
    enhanced_prompt = " ".join([result.page_content for result in results])
    logger.info("Enhanced prompt created successfully!")
    return enhanced_prompt

# Initialize the knowledge base
def initialize_rag():
    pdf_path = "/Users/SathwikReddyChelemela/Documents/finalprompt project/TextToVideo-fork/Backend/flickr8k_knowledge_base.pdf"  # Replace with your actual PDF file
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    vector_store = create_knowledge_base(pdf_path)
    return vector_store
