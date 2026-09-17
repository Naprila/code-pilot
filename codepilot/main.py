from dotenv import load_dotenv
from codepilot.observability.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

def run():
    logger.info("Hello from codepilot")