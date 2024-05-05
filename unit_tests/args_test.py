import pytest
from internal_controlnet.args import ControlNetUnit


@pytest.fixture(scope="module")
def set_cls_funcs():
    ControlNetUnit.cls_match_model = lambda s: s in {"None", "model1", "model2"}
    ControlNetUnit.cls_match_module = lambda s: s in {"none", "module1"}


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
