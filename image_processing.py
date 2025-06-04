import os
import io
import base64
import uuid
from PIL import Image

# Define the default directory for image outputs relative to this file's location
_default_image_output_dir = os.path.join(os.path.dirname(__file__), "image_outputs")

# Helper functions for image processing
def encode_image(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def decode_image(base64_string: str) -> Image.Image:
    """Convert a base64 string to a PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def save_image(image: Image.Image, directory: str = None) -> str:
    """Save a PIL Image to disk and return the path."""
    if directory is None:
        directory = _default_image_output_dir
    os.makedirs(directory, exist_ok=True)
    image_id = str(uuid.uuid4())
    image_path = os.path.join(directory, f"{image_id}.png")
    image.save(image_path)
    return image_path
