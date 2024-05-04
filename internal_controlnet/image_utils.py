import numpy as np

from modules.api import api


def to_base64_nparray(encoding: str) -> np.ndarray:
    """
    Convert a base64 image into the image type the extension uses
    """

    return np.array(api.decode_base64_to_image(encoding)).astype("uint8")
