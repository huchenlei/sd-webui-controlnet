from copy import copy
from typing import List, Any, Optional, Union, Tuple, Dict

from modules import scripts, processing, shared
from scripts import global_state
from scripts.logging import logger
from scripts.enums import ResizeMode
from scripts.supported_preprocessor import Preprocessor
from .args import ControlNetUnit


resize_mode_aliases = {
    "Inner Fit (Scale to Fit)": "Crop and Resize",
    "Outer Fit (Shrink to Fit)": "Resize and Fill",
    "Scale to Fit (Inner Fit)": "Crop and Resize",
    "Envelope (Outer Fit)": "Resize and Fill",
}


def resize_mode_from_value(value: Union[str, int, ResizeMode]) -> ResizeMode:
    if isinstance(value, str):
        return ResizeMode(resize_mode_aliases.get(value, value))
    elif isinstance(value, int):
        assert value >= 0
        if value == 3:  # 'Just Resize (Latent upscale)'
            return ResizeMode.RESIZE

        if value >= len(ResizeMode):
            logger.warning(
                f"Unrecognized ResizeMode int value {value}. Fall back to RESIZE."
            )
            return ResizeMode.RESIZE

        return [e for e in ResizeMode][value]
    else:
        return value


def get_all_units_in_processing(
    p: processing.StableDiffusionProcessing,
) -> List[ControlNetUnit]:
    """
    Fetch ControlNet processing units from a StableDiffusionProcessing.
    """

    return get_all_units(p.scripts, p.script_args)


def get_all_units(
    script_runner: scripts.ScriptRunner, script_args: List[Any]
) -> List[ControlNetUnit]:
    """
    Fetch ControlNet processing units from an existing script runner.
    Use this function to fetch units from the list of all scripts arguments.
    """

    cn_script = find_cn_script(script_runner)
    if cn_script:
        return get_all_units_from(script_args[cn_script.args_from : cn_script.args_to])

    return []


def get_all_units_from(script_args: List[Any]) -> List[ControlNetUnit]:
    """
    Fetch ControlNet processing units from ControlNet script arguments.
    Use `external_code.get_all_units` to fetch units from the list of all scripts arguments.
    """
    all_units = [to_processing_unit(script_arg) for script_arg in script_args]
    return all_units


def get_single_unit_from(
    script_args: List[Any], index: int = 0
) -> Optional[ControlNetUnit]:
    """
    Fetch a single ControlNet processing unit from ControlNet script arguments.
    The list must not contain script positional arguments. It must only contain processing units.
    """

    i = 0
    while i < len(script_args) and index >= 0:
        if index == 0 and script_args[i] is not None:
            return to_processing_unit(script_args[i])
        i += 1

        index -= 1

    return None


def get_max_models_num():
    """
    Fetch the maximum number of allowed ControlNet models.
    """

    max_models_num = shared.opts.data.get("control_net_unit_count", 3)
    return max_models_num


def to_processing_unit(unit: Dict[str, Any]) -> ControlNetUnit:
    """
    Convert dict to processing unit.
    """
    return ControlNetUnit(**unit)


def update_cn_script_in_processing(
    p: processing.StableDiffusionProcessing,
    cn_units: List[ControlNetUnit],
    **_kwargs,  # for backwards compatibility
):
    """
    Update the arguments of the ControlNet script in `p.script_args` in place, reading from `cn_units`.
    `cn_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.

    Does not update `p.script_args` if any of the folling is true:
    - ControlNet is not present in `p.scripts`
    - `p.script_args` is not filled with script arguments for scripts that are processed before ControlNet
    """
    p.script_args = update_cn_script(p.scripts, p.script_args_value, cn_units)


def update_cn_script(
    script_runner: scripts.ScriptRunner,
    script_args: Union[Tuple[Any], List[Any]],
    cn_units: List[ControlNetUnit],
) -> Union[Tuple[Any], List[Any]]:
    """
    Returns: The updated `script_args` with given `cn_units` used as ControlNet
    script args.

    Does not update `script_args` if any of the folling is true:
    - ControlNet is not present in `script_runner`
    - `script_args` is not filled with script arguments for scripts that are
    processed before ControlNet
    """
    script_args_type = type(script_args)
    assert script_args_type in (tuple, list), script_args_type
    updated_script_args = list(copy(script_args))

    cn_script = find_cn_script(script_runner)

    if cn_script is None or len(script_args) < cn_script.args_from:
        return script_args

    # fill in remaining parameters to satisfy max models, just in case script needs it.
    max_models = shared.opts.data.get("control_net_unit_count", 3)
    cn_units = cn_units + [ControlNetUnit(enabled=False)] * max(
        max_models - len(cn_units), 0
    )

    cn_script_args_diff = 0
    for script in script_runner.alwayson_scripts:
        if script is cn_script:
            cn_script_args_diff = len(cn_units) - (
                cn_script.args_to - cn_script.args_from
            )
            updated_script_args[script.args_from : script.args_to] = cn_units
            script.args_to = script.args_from + len(cn_units)
        else:
            script.args_from += cn_script_args_diff
            script.args_to += cn_script_args_diff

    return script_args_type(updated_script_args)


def update_cn_script_in_place(
    script_runner: scripts.ScriptRunner,
    script_args: List[Any],
    cn_units: List[ControlNetUnit],
    **_kwargs,  # for backwards compatibility
):
    """
    @Deprecated(Raises assertion error if script_args passed in is Tuple)
    Use `update_cn_script` instead

    Update the arguments of the ControlNet script in `script_args` in place, reading from `cn_units`.
    `cn_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.

    Does not update `script_args` if any of the folling is true:
    - ControlNet is not present in `script_runner`
    - `script_args` is not filled with script arguments for scripts that are processed before ControlNet
    """
    assert isinstance(script_args, list), type(script_args)

    cn_script = find_cn_script(script_runner)
    if cn_script is None or len(script_args) < cn_script.args_from:
        return

    # fill in remaining parameters to satisfy max models, just in case script needs it.
    max_models = get_max_models_num()
    cn_units = cn_units + [ControlNetUnit(enabled=False)] * max(
        max_models - len(cn_units), 0
    )

    cn_script_args_diff = 0
    for script in script_runner.alwayson_scripts:
        if script is cn_script:
            cn_script_args_diff = len(cn_units) - (
                cn_script.args_to - cn_script.args_from
            )
            script_args[script.args_from : script.args_to] = cn_units
            script.args_to = script.args_from + len(cn_units)
        else:
            script.args_from += cn_script_args_diff
            script.args_to += cn_script_args_diff


def get_models(update: bool = False) -> List[str]:
    """
    Fetch the list of available models.
    Each value is a valid candidate of `ControlNetUnit.model`.

    Keyword arguments:
    update -- Whether to refresh the list from disk. (default False)
    """

    if update:
        global_state.update_cn_models()

    return list(global_state.cn_models_names.values())


def get_modules(alias_names: bool = False) -> List[str]:
    """
    Fetch the list of available preprocessors.
    Each value is a valid candidate of `ControlNetUnit.module`.

    Keyword arguments:
    alias_names -- Whether to get the ui alias names instead of internal keys
    """
    return [
        (p.label if alias_names else p.name)
        for p in Preprocessor.get_sorted_preprocessors()
    ]


def get_modules_detail(alias_names: bool = False) -> Dict[str, Any]:
    """
    get the detail of all preprocessors including
    sliders: the slider config in Auto1111 webUI

    Keyword arguments:
    alias_names -- Whether to get the module detail with alias names instead of internal keys
    """

    _module_detail = {}
    _module_list = get_modules(False)
    _module_list_alias = get_modules(True)

    _output_list = _module_list if not alias_names else _module_list_alias
    for module_name in _output_list:
        preprocessor = Preprocessor.get_preprocessor(module_name)
        assert preprocessor is not None
        _module_detail[module_name] = dict(
            model_free=preprocessor.do_not_need_model,
            sliders=[
                s.api_json
                for s in (
                    preprocessor.slider_resolution,
                    preprocessor.slider_1,
                    preprocessor.slider_2,
                    preprocessor.slider_3,
                )
                if s.visible
            ],
        )

    return _module_detail


def find_cn_script(script_runner: scripts.ScriptRunner) -> Optional[scripts.Script]:
    """
    Find the ControlNet script in `script_runner`. Returns `None` if `script_runner` does not contain a ControlNet script.
    """

    if script_runner is None:
        return None

    for script in script_runner.alwayson_scripts:
        if is_cn_script(script):
            return script


def is_cn_script(script: scripts.Script) -> bool:
    """
    Determine whether `script` is a ControlNet script.
    """

    return script.title().lower() == "controlnet"
