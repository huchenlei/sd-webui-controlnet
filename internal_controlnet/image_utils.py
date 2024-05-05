import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from typing import Optional

from modules.api import api
from modules.processing import StableDiffusionProcessing

from scripts.enums import ResizeMode
from scripts.logging import logger


def to_base64_nparray(encoding: str) -> np.ndarray:
    """
    Convert a base64 image into the image type the extension uses
    """

    return np.array(api.decode_base64_to_image(encoding)).astype("uint8")


def visualize_inpaint_mask(img):
    if img.ndim == 3 and img.shape[2] == 4:
        result = img.copy()
        mask = result[:, :, 3]
        mask = 255 - mask // 2
        result[:, :, 3] = mask
        return np.ascontiguousarray(result.copy())
    return img


def pixel_perfect_resolution(
    image: np.ndarray,
    target_H: int,
    target_W: int,
    resize_mode: ResizeMode,
) -> int:
    """
    Calculate the estimated resolution for resizing an image while preserving aspect ratio.

    The function first calculates scaling factors for height and width of the image based on the target
    height and width. Then, based on the chosen resize mode, it either takes the smaller or the larger
    scaling factor to estimate the new resolution.

    If the resize mode is OUTER_FIT, the function uses the smaller scaling factor, ensuring the whole image
    fits within the target dimensions, potentially leaving some empty space.

    If the resize mode is not OUTER_FIT, the function uses the larger scaling factor, ensuring the target
    dimensions are fully filled, potentially cropping the image.

    After calculating the estimated resolution, the function prints some debugging information.

    Args:
        image (np.ndarray): A 3D numpy array representing an image. The dimensions represent [height, width, channels].
        target_H (int): The target height for the image.
        target_W (int): The target width for the image.
        resize_mode (ResizeMode): The mode for resizing.

    Returns:
        int: The estimated resolution after resizing.
    """
    raw_H, raw_W, _ = image.shape

    k0 = float(target_H) / float(raw_H)
    k1 = float(target_W) / float(raw_W)

    if resize_mode == ResizeMode.OUTER_FIT:
        estimation = min(k0, k1) * float(min(raw_H, raw_W))
    else:
        estimation = max(k0, k1) * float(min(raw_H, raw_W))

    logger.debug("Pixel Perfect Computation:")
    logger.debug(f"resize_mode = {resize_mode}")
    logger.debug(f"raw_H = {raw_H}")
    logger.debug(f"raw_W = {raw_W}")
    logger.debug(f"target_H = {target_H}")
    logger.debug(f"target_W = {target_W}")
    logger.debug(f"estimation = {estimation}")

    return int(np.round(estimation))


def prepare_mask(mask: Image.Image, p: StableDiffusionProcessing) -> Image.Image:
    """
    Prepare an image mask for the inpainting process.

    This function takes as input a PIL Image object and an instance of the
    StableDiffusionProcessing class, and performs the following steps to prepare the mask:

    1. Convert the mask to grayscale (mode "L").
    2. If the 'inpainting_mask_invert' attribute of the processing instance is True,
       invert the mask colors.
    3. If the 'mask_blur' attribute of the processing instance is greater than 0,
       apply a Gaussian blur to the mask with a radius equal to 'mask_blur'.

    Args:
        mask (Image.Image): The input mask as a PIL Image object.
        p (processing.StableDiffusionProcessing): An instance of the StableDiffusionProcessing class
                                                   containing the processing parameters.

    Returns:
        mask (Image.Image): The prepared mask as a PIL Image object.
    """
    mask = mask.convert("L")
    if getattr(p, "inpainting_mask_invert", False):
        mask = ImageOps.invert(mask)

    if hasattr(p, "mask_blur_x"):
        if getattr(p, "mask_blur_x", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_x + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), p.mask_blur_x)
            mask = Image.fromarray(np_mask)
        if getattr(p, "mask_blur_y", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_y + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), p.mask_blur_y)
            mask = Image.fromarray(np_mask)
    else:
        if getattr(p, "mask_blur", 0) > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    return mask


def set_numpy_seed(p: StableDiffusionProcessing) -> Optional[int]:
    """
    Set the random seed for NumPy based on the provided parameters.

    Args:
        p (processing.StableDiffusionProcessing): The instance of the StableDiffusionProcessing class.

    Returns:
        Optional[int]: The computed random seed if successful, or None if an exception occurs.

    This function sets the random seed for NumPy using the seed and subseed values from the given instance of
    StableDiffusionProcessing. If either seed or subseed is -1, it uses the first value from `all_seeds`.
    Otherwise, it takes the maximum of the provided seed value and 0.

    The final random seed is computed by adding the seed and subseed values, applying a bitwise AND operation
    with 0xFFFFFFFF to ensure it fits within a 32-bit integer.
    """
    try:
        tmp_seed = int(p.all_seeds[0] if p.seed == -1 else max(int(p.seed), 0))
        tmp_subseed = int(p.all_seeds[0] if p.subseed == -1 else max(int(p.subseed), 0))
        seed = (tmp_seed + tmp_subseed) & 0xFFFFFFFF
        np.random.seed(seed)
        return seed
    except Exception as e:
        logger.warning(e)
        logger.warning("Warning: Failed to use consistent random seed.")
        return None
