# __init__.py for the app package

# Import the main components to make them available when the package is imported
from .cli import parse_args
from .config import *
from .encoder import Encoder
from .decoder import Decoder
from .data_handler import load_csv

# Optional: Setup logging or other package-wide configurations here
import logging

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# You can also include any initialization code that should be executed when the package is imported
