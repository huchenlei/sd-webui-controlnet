import os
import torch
import numpy as np
from typing import Optional, List, Annotated, ClassVar, Callable, Any
from pydantic import BaseModel, validator, root_validator, Field
from PIL import Image

from scripts.enums import (
    ResizeMode,
    ControlMode,
    HiResFixOption,
    PuLIDMode,
)
from scripts.logging import logger


def _unimplemented_func(*args, **kwargs):
    raise NotImplementedError("Not implemented.")


class ControlNetUnit(BaseModel):
    """
    Represents an entire ControlNet processing unit.
    """

    class Config:
        arbitrary_types_allowed = True

    cls_match_module: ClassVar[Callable[[str], bool]] = _unimplemented_func
    cls_match_model: ClassVar[Callable[[str], bool]] = _unimplemented_func
    cls_decode_base64: ClassVar[Callable[[str], np.ndarray]] = _unimplemented_func
    cls_torch_load_base64: ClassVar[Callable[[Any], torch.Tensor]] = _unimplemented_func
    cls_get_preprocessor: ClassVar[Callable[[str], Any]] = _unimplemented_func

    enabled: bool = True
    module: str = "none"

    @validator("module", always=True, pre=True)
    def check_module(cls, value: str) -> str:
        if not ControlNetUnit.cls_match_module(value):
            raise ValueError(f"module({value}) not found in supported modules.")
        return value

    model: str = "None"

    @validator("model", always=True, pre=True)
    def check_model(cls, value: str) -> str:
        if not ControlNetUnit.cls_match_model(value):
            raise ValueError(f"model({value}) not found in supported models.")
        return value

    weight: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
    # [B, H, W, 4] RGBA
    # Optional[np.ndarray]
    image: Any = None

    resize_mode: ResizeMode = ResizeMode.INNER_FIT
    low_vram: bool = False
    processor_res: int = -1
    threshold_a: float = -1
    threshold_b: float = -1

    @root_validator
    def bound_check_params(cls, values: dict) -> dict:
        """
        Checks and corrects negative parameters in ControlNetUnit 'unit' in place.
        Parameters 'processor_res', 'threshold_a', 'threshold_b' are reset to
        their default values if negative.
        """
        module = values.get("module")
        if not module:
            return values

        preprocessor = cls.cls_get_preprocessor(module)
        assert preprocessor is not None
        for unit_param, param in zip(
            ("processor_res", "threshold_a", "threshold_b"),
            ("slider_resolution", "slider_1", "slider_2"),
        ):
            value = values.get(unit_param)
            cfg = getattr(preprocessor, param)
            if value < cfg.minimum or value > cfg.maximum:
                values[unit_param] = cfg.value
                logger.info(
                    f"[{module}.{unit_param}] Invalid value({value}), using default value {cfg.value}."
                )
        return values

    guidance_start: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    guidance_end: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    @root_validator
    def guidance_check(cls, values: dict) -> dict:
        start = values.get("guidance_start")
        end = values.get("guidance_end")
        if start > end:
            raise ValueError(f"guidance_start({start}) > guidance_end({end})")
        return values

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

    @validator("effective_region_mask", pre=True)
    def parse_effective_region_mask(cls, value) -> np.ndarray:
        if isinstance(value, str):
            return cls.cls_decode_base64(value)
        assert isinstance(value, np.ndarray) or value is None
        return value

    # The weight mode for PuLID.
    # https://github.com/ToTheBeginning/PuLID
    pulid_mode: PuLIDMode = PuLIDMode.FIDELITY

    # ------- API only fields -------
    # The tensor input for ipadapter. When this field is set in the API,
    # the base64string will be interpret by torch.load to reconstruct ipadapter
    # preprocessor output.
    # Currently the option is only accessible in API calls.
    ipadapter_input: Optional[List[torch.Tensor]] = None

    @validator("ipadapter_input", pre=True)
    def parse_ipadapter_input(cls, value) -> Optional[List[torch.Tensor]]:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        result = [cls.cls_torch_load_base64(b) for b in value]
        assert result, "input cannot be empty"
        return result

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
    def parse_image_formats(cls, values: dict) -> dict:
        """
        Parse image with following formats.
        API
        - image = {"image": base64image, "mask": base64image,}
        - image = [image, mask]
        - image = (image, mask)
        - image = [{"image": ..., "mask": ...}, {"image": ..., "mask": ...}, ...]
        - image = base64image, mask = base64image

        UI
        - image = np.ndarray (B, H, W, 4)
        """
        init_image = values.get("image")
        init_mask = values.get("mask")

        if init_image is None:
            assert init_mask is None
            return values

        if isinstance(init_image, np.ndarray):
            assert init_image.ndim == 4
            assert init_image.shape[-1] == 4
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
        elif isinstance(init_image, str):
            # image = base64image, mask = base64image
            images = [
                {
                    "image": init_image,
                    "mask": init_mask,
                }
            ]
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
                    return cls.cls_decode_base64(image)

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
            )[:, :, 0:1]
            np_images.append(np.concatenate([np_image, np_mask], axis=2))  # [H, W, 4]

        final_np_image = np.stack(np_images, axis=0)  # [B, H, W, 4]
        assert final_np_image.ndim == 4
        assert final_np_image.shape[-1] == 4
        values["image"] = final_np_image

        return values

    @staticmethod
    def infotext_excluded_fields() -> List[str]:
        return [
            "image",
            "enabled",
            "advanced_weighting",
            "ipadapter_input",
            # Note: "inpaint_crop_image" is img2img inpaint only flag, which does not
            # provide much information when restoring the unit.
            "inpaint_crop_input_image",
            "effective_region_mask",
            "pulid_mode",
        ]

    @property
    def accepts_multiple_inputs(self) -> bool:
        """This unit can accept multiple input images."""
        return self.module in (
            "ip-adapter-auto",
            "ip-adapter_clip_sdxl",
            "ip-adapter_clip_sdxl_plus_vith",
            "ip-adapter_clip_sd15",
            "ip-adapter_face_id",
            "ip-adapter_face_id_plus",
            "ip-adapter_pulid",
            "instant_id_face_embedding",
        )

    @property
    def is_animate_diff_batch(self) -> bool:
        return getattr(self, "animatediff_batch", False)

    @property
    def uses_clip(self) -> bool:
        """Whether this unit uses clip preprocessor."""
        return any(
            (
                ("ip-adapter" in self.module and "face_id" not in self.module),
                self.module
                in ("clip_vision", "revision_clipvision", "revision_ignore_prompt"),
            )
        )

    @property
    def is_inpaint(self) -> bool:
        return "inpaint" in self.module
