import os
import torch
import numpy as np
from typing import Optional, List, Annotated
from pydantic import BaseModel, validator, root_validator, Field
from PIL import Image

from scripts.enums import (
    ResizeMode,
    ControlMode,
    HiResFixOption,
    PuLIDMode,
)
from scripts.supported_preprocessor import Preprocessor
from scripts.logging import logger
from .image_utils import to_base64_nparray


class ControlNetUnit(BaseModel):
    """
    Represents an entire ControlNet processing unit.
    """

    enabled: bool = True
    module: str = "none"

    @validator("module", always=True, pre=True)
    def check_module(cls, value: str) -> str:
        p = Preprocessor.get_preprocessor(value)
        if p is None:
            raise ValueError(f"module({value}) not found in supported modules.")
        return value

    # TODO: Validate model.
    model: str = "None"
    weight: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
    # [B, H, W, 4] RGBA
    image: Optional[np.ndarray] = None

    resize_mode: ResizeMode = ResizeMode.INNER_FIT
    low_vram: bool = False
    processor_res: int = -1
    threshold_a: float = -1
    threshold_b: float = -1
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    pixel_perfect: bool = False
    control_mode: ControlMode = ControlMode.BALANCED
    # Whether to crop input image based on A1111 img2img mask. This flag is only used when `inpaint area`
    # in A1111 is set to `Only masked`. In API, this correspond to `inpaint_full_res = True`.
    inpaint_crop_input_image: bool = True
    # If hires fix is enabled in A1111, how should this ControlNet unit be applied.
    # The value is ignored if the generation is not using hires fix.
    hr_option: HiResFixOption = HiResFixOption.BOTH

    # Whether save the detected map of this unit. Setting this option to False prevents saving the
    # detected map or sending detected map along with generated images via API.
    # Currently the option is only accessible in API calls.
    save_detected_map: bool = True

    # Weight for each layer of ControlNet params.
    # For ControlNet:
    # - SD1.5: 13 weights (4 encoder block * 3 + 1 middle block)
    # - SDXL: 10 weights (3 encoder block * 3 + 1 middle block)
    # For T2IAdapter
    # - SD1.5: 5 weights (4 encoder block + 1 middle block)
    # - SDXL: 4 weights (3 encoder block + 1 middle block)
    # For IPAdapter
    # - SD15: 16 (6 input blocks + 9 output blocks + 1 middle block)
    # - SDXL: 11 weights (4 input blocks + 6 output blocks + 1 middle block)
    # Note1: Setting advanced weighting will disable `soft_injection`, i.e.
    # It is recommended to set ControlMode = BALANCED when using `advanced_weighting`.
    # Note2: The field `weight` is still used in some places, e.g. reference_only,
    # even advanced_weighting is set.
    advanced_weighting: Optional[List[float]] = None

    # The effective region mask that unit's effect should be restricted to.
    effective_region_mask: Optional[np.ndarray] = None

    # The weight mode for PuLID.
    # https://github.com/ToTheBeginning/PuLID
    pulid_mode: PuLIDMode = PuLIDMode.FIDELITY

    # ------- API only fields -------
    # The tensor input for ipadapter. When this field is set in the API,
    # the base64string will be interpret by torch.load to reconstruct ipadapter
    # preprocessor output.
    # Currently the option is only accessible in API calls.
    ipadapter_input: Optional[List[torch.Tensor]] = None

    mask: Optional[str] = None
    mask_image: Optional[str] = None

    @root_validator
    def mask_alias(cls, values: dict) -> dict:
        """
        Field "mask_image" is the alias of field "mask".
        This is for compatibility with SD Forge API.
        """
        mask_image = values.get("mask_image")
        mask = values.get("mask")
        if mask_image is not None:
            if mask is not None:
                raise ValueError("Cannot specify both 'mask' and 'mask_image'!")
            values["mask"] = mask_image
        return values

    @root_validator
    def parse_legacy_image_formats(cls, values: dict) -> dict:
        """
        Parse image with following legacy formats.
        - {"image": ..., "mask": ...}
        - [image, mask]
        - (image, mask)
        - [{"image": ..., "mask": ...}, {"image": ..., "mask": ...}, ...]
        """
        init_image = values.get("image")
        if init_image is None or isinstance(init_image, np.ndarray):
            return values

        if isinstance(init_image, (list, tuple)):
            if not init_image:
                raise ValueError(f"{init_image} is not a valid 'image' field value")
            if isinstance(init_image[0], dict):
                # [{"image": ..., "mask": ...}, {"image": ..., "mask": ...}, ...]
                images = init_image
            else:
                assert len(init_image) == 2
                # [image, mask]
                # (image, mask)
                images = [
                    {
                        "image": init_image[0],
                        "mask": init_image[1],
                    }
                ]
        elif isinstance(init_image, dict):
            # {"image": ..., "mask": ...}
            images = [init_image]
        else:
            raise ValueError(f"Unrecognized image field {init_image}")

        def parse_image(image) -> np.ndarray:
            if isinstance(image, np.ndarray):
                return image

            if isinstance(image, str):
                if os.path.exists(image):
                    logger.warn(
                        "Reading image from local disk will be deprecated 2024-06-01."
                    )
                    return np.array(Image.open(image["image"])).astype("uint8")
                else:
                    return to_base64_nparray(image)

            raise ValueError(f"Unrecognized image format {image}.")

        np_images = []
        for image_dict in images:
            assert isinstance(image_dict, dict)
            image = image_dict.get("image")
            mask = image_dict.get("mask")
            assert image is not None

            np_image = parse_image(image)
            np_mask = (
                np.ones_like(np_image) * 255 if mask is None else parse_image(mask)
            )
            np_images.append(np.concatenate([np_image, np_mask], axis=2))  # [H, W, 4]

        final_np_image = np.stack(np_images, axis=0)  # [B, H, W, 4]
        assert final_np_image.ndim == 4
        assert final_np_image.shape[-1] == 4
        values["image"] = final_np_image

        return values
