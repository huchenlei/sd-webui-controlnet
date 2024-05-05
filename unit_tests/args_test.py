import pytest
import numpy as np

from internal_controlnet.args import ControlNetUnit

H = W = 128

img1 = np.ones(shape=[H, W, 3], dtype=np.uint8)
img2 = np.ones(shape=[H, W, 3], dtype=np.uint8) * 2
ui_img = np.ones(shape=[1, H, W, 4], dtype=np.uint8)


@pytest.fixture(scope="module")
def set_cls_funcs():
    ControlNetUnit.cls_match_model = lambda s: s in {"None", "model1", "model2"}
    ControlNetUnit.cls_match_module = lambda s: s in {"none", "module1"}
    ControlNetUnit.cls_decode_base64 = lambda s: {
        "b64img1": img1,
        "b64img2": img2,
    }[s]


def test_module_invalid(set_cls_funcs):
    with pytest.raises(ValueError) as excinfo:
        ControlNetUnit(module="foo")

    assert "module(foo) not found in supported modules." in str(excinfo.value)


def test_module_valid(set_cls_funcs):
    ControlNetUnit(module="module1")


def test_model_invalid(set_cls_funcs):
    with pytest.raises(ValueError) as excinfo:
        ControlNetUnit(model="foo")

    assert "model(foo) not found in supported models." in str(excinfo.value)


def test_model_valid(set_cls_funcs):
    ControlNetUnit(model="model1")


def test_valid_image_formats(set_cls_funcs):
    ControlNetUnit(image={"image": "b64img1"})
    ControlNetUnit(image={"image": "b64img1", "mask": "b64img2"})
    ControlNetUnit(image=["b64img1", "b64img2"])
    ControlNetUnit(image=("b64img1", "b64img2"))
    ControlNetUnit(image=[{"image": "b64img1", "mask": "b64img2"}])
    ControlNetUnit(image=[{"image": "b64img1"}])
    ControlNetUnit(image=[{"image": "b64img1", "mask": None}])
    ControlNetUnit(
        image=[
            {"image": "b64img1", "mask": "b64img2"},
            {"image": "b64img1", "mask": "b64img2"},
        ]
    )
    ControlNetUnit(
        image=[
            {"image": "b64img1", "mask": None},
            {"image": "b64img1", "mask": "b64img2"},
        ]
    )
    ControlNetUnit(
        image=[
            {"image": "b64img1"},
            {"image": "b64img1", "mask": "b64img2"},
        ]
    )
    ControlNetUnit(image="b64img1", mask="b64img2")
    ControlNetUnit(image="b64img1")
    ControlNetUnit(image="b64img1", mask_image="b64img2")
    ControlNetUnit(image=ui_img)
    ControlNetUnit(image=None)


@pytest.mark.parametrize(
    "d",
    [
        dict(image={"mask": "b64img1"}),
        dict(image={"foo": "b64img1", "bar": "b64img2"}),
        dict(image=["b64img1"]),
        dict(image=("b64img1", "b64img2", "b64img1")),
        dict(image=[]),
        dict(image=[{"mask": "b64img1"}]),
        dict(image=None, mask="b64img2"),
        dict(image=img1),  # Wrong shape
        dict(image="b64img1", mask="b64img1", mask_image="b64img1"),
    ],
)
def test_invalid_image_formats(set_cls_funcs, d):
    with pytest.raises(ValueError):
        ControlNetUnit(**d)
