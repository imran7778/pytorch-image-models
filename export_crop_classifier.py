#!/usr/bin/env python3
"""Export the trained crop classifier to ONNX and TensorRT from one YAML."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import timm


INTERPOLATION = re.compile(r"\$\{([^}]+)}")


def load_config(path: Path) -> dict[str, Any]:
    with path.resolve().open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return resolve_tree(cfg, cfg)


def resolve_tree(value: Any, root: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: resolve_tree(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_tree(item, root) for item in value]
    if not isinstance(value, str):
        return value
    match = INTERPOLATION.fullmatch(value)
    if match:
        return resolve_tree(lookup(root, match.group(1)), root)
    result = value
    for _ in range(10):
        updated = INTERPOLATION.sub(lambda item: str(lookup(root, item.group(1))), result)
        if updated == result:
            return updated
        result = updated
    raise ValueError(f"Interpolation did not converge: {value}")


def lookup(root: dict[str, Any], dotted_key: str) -> Any:
    value: Any = root
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Unknown interpolation ${{{dotted_key}}}")
        value = value[part]
    return value


def load_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device):
    model_cfg = cfg["model"]
    model = timm.create_model(
        str(model_cfg["name"]),
        pretrained=False,
        num_classes=int(model_cfg["num_classes"]),
        in_chans=3,
        exportable=True,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {checkpoint}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    return model, payload


def export_onnx(
    cfg: dict[str, Any],
    model,
    checkpoint: Path,
    output: Path,
    device: torch.device,
) -> Path:
    export_cfg = cfg["export"]
    common = export_cfg["common_args"]
    onnx_cfg = export_cfg["onnx_args"]
    image_size = normalize_size(common["input_size"])
    batch = int(onnx_cfg.get("batch", 1))
    precision = str(onnx_cfg["precision"]).lower()
    if precision not in {"fp32", "fp16"}:
        raise ValueError("export.onnx_args.precision must be fp32 or fp16")
    # Export ConvNeXt in FP32 first. Direct model.half() export leaves
    # LayerNormalization with mixed FP16/FP32 inputs. Graph-aware conversion
    # below converts weights and constants consistently.
    model = model.float()
    dtype = torch.float32
    dummy = torch.zeros(
        batch,
        int(common.get("input_channels", 3)),
        image_size[1],
        image_size[0],
        device=device,
        dtype=dtype,
    )
    input_name = str(onnx_cfg.get("input_name", "images"))
    output_name = str(onnx_cfg.get("output_name", "logits"))
    dynamic_axes = None
    if bool(common.get("dynamic_batch", False)) or bool(common.get("dynamic_spatial", False)):
        input_axes: dict[int, str] = {}
        output_axes: dict[int, str] = {}
        if bool(common.get("dynamic_batch", False)):
            input_axes[0] = "batch"
            output_axes[0] = "batch"
        if bool(common.get("dynamic_spatial", False)):
            input_axes.update({2: "height", 3: "width"})
        dynamic_axes = {input_name: input_axes, output_name: output_axes}

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX: {checkpoint} -> {output}", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=[input_name],
            output_names=[output_name],
            opset_version=int(common.get("opset", 18)),
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
            verbose=bool(common.get("verbose", False)),
        )
    if precision == "fp16":
        convert_onnx_to_fp16(output)
    _check_and_annotate_onnx(output, cfg, precision)
    if bool(common.get("simplify", True)):
        simplify_onnx(output)
    if bool(onnx_cfg.get("verify", True)):
        verify_onnx(output, model, dummy, onnx_cfg)
    return output


def _check_and_annotate_onnx(path: Path, cfg: dict[str, Any], precision: str) -> None:
    try:
        import onnx
    except ImportError as error:
        raise ImportError("ONNX export validation requires: pip install onnx") from error
    graph = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(graph)
    metadata = {
        "model": str(cfg["model"]["name"]),
        "num_classes": str(cfg["model"]["num_classes"]),
        "names": json.dumps(["drone", "bird", "airplane", "helicopter"]),
        "input_size": json.dumps(normalize_size(cfg["export"]["common_args"]["input_size"])),
        "precision": precision,
        "crop_expansion": str(cfg["data"]["crop_expansion"]),
        "mean": json.dumps([0.485, 0.456, 0.406]),
        "std": json.dumps([0.229, 0.224, 0.225]),
    }
    del graph.metadata_props[:]
    for key, value in metadata.items():
        entry = graph.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(graph, str(path))


def simplify_onnx(path: Path) -> None:
    try:
        import onnx
        import onnxslim
    except ImportError:
        print("WARNING: simplify=true but onnxslim is unavailable; keeping checked ONNX graph.", flush=True)
        return
    graph = onnx.load(str(path), load_external_data=True)
    simplified = onnxslim.slim(graph)
    onnx.save(simplified, str(path))
    print("ONNX simplification: passed", flush=True)


def convert_onnx_to_fp16(path: Path) -> None:
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as error:
        raise ImportError("FP16 ONNX conversion requires: pip install onnxconverter-common") from error
    graph = onnx.load(str(path), load_external_data=True)
    converted = float16.convert_float_to_float16(
        graph,
        keep_io_types=False,
        disable_shape_infer=False,
    )
    onnx.save(converted, str(path))
    print("ONNX graph-aware FP16 conversion: passed", flush=True)


def verify_onnx(path: Path, model, dummy: torch.Tensor, options: dict[str, Any]) -> None:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError("ONNX verification requires onnxruntime or onnxruntime-gpu") from error
    requested = str(options.get("provider", "cuda")).lower()
    available = set(ort.get_available_providers())
    providers = []
    if requested == "cuda" and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(path), providers=providers or None)
    active = session.get_providers()
    if bool(options.get("require_gpu", False)) and "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"ONNX GPU verification required, but active providers are {active}")
    with torch.inference_mode():
        expected = model(dummy).detach().float().cpu().numpy()
    input_info = session.get_inputs()[0]
    input_array = dummy.detach().cpu().numpy()
    if "float16" in input_info.type:
        input_array = input_array.astype(np.float16)
    actual = session.run(None, {input_info.name: input_array})[0].astype(np.float32)
    np.testing.assert_allclose(
        actual,
        expected,
        atol=float(options.get("atol", 1e-3)),
        rtol=float(options.get("rtol", 1e-2)),
    )
    print(f"ONNX verification: passed with providers={active}, output={actual.shape}", flush=True)


def export_engine(cfg: dict[str, Any], onnx_path: Path, output: Path) -> Path:
    options = cfg["export"]["engine_args"]
    common = cfg["export"]["common_args"]
    try:
        import tensorrt as trt
    except ImportError as error:
        raise ImportError("TensorRT export requires the tensorrt Python package") from error
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT export requires a working CUDA GPU")
    logger = trt.Logger(trt.Logger.INFO if bool(common.get("verbose", False)) else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    explicit = getattr(getattr(trt, "NetworkDefinitionCreationFlag", object), "EXPLICIT_BATCH", None)
    if explicit is not None:
        flags |= 1 << int(explicit)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        messages = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{messages}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(float(options.get("workspace_gb", 4)) * (1 << 30)),
    )
    optimization_level = int(options.get("builder_optimization_level", 5))
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = optimization_level
    precision = str(options.get("precision", "fp16")).lower()
    if precision == "fp16":
        # TensorRT 10+ removed platform_has_fast_fp16; absence means the
        # graph precision is authoritative.
        if not getattr(builder, "platform_has_fast_fp16", True):
            print("WARNING: target reports no fast FP16 support.", flush=True)
        fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
        if fp16_flag is not None:
            config.set_flag(fp16_flag)
        else:
            print("TensorRT has no FP16 builder flag; using the ONNX graph's FP16 types.", flush=True)
    elif precision == "int8":
        raise NotImplementedError("INT8 engine export requires representative classifier calibration data")
    elif precision != "fp32":
        raise ValueError("export.engine_args.precision must be fp32, fp16, or int8")
    _configure_trt_profile(network, builder, config, common, options)
    _configure_trt_verbosity(config, trt, options)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building TensorRT {precision.upper()} engine: {onnx_path} -> {output}", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    output.write_bytes(serialized)
    if bool(options.get("verify", True)):
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(output.read_bytes())
        if engine is None:
            raise RuntimeError("TensorRT verification failed: engine cannot be deserialized")
        input_names = [
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
        ]
        output_names = [
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
        ]
        print(f"TensorRT verification: passed, inputs={input_names}, outputs={output_names}", flush=True)
    return output


def _configure_trt_profile(network, builder, config, common: dict[str, Any], options: dict[str, Any]) -> None:
    if not bool(common.get("dynamic_batch", False)):
        return
    input_tensor = network.get_input(0)
    shape = list(input_tensor.shape)
    if shape[0] >= 0:
        return
    profile = builder.create_optimization_profile()
    minimum = tuple([int(options.get("min_batch", 1)), *shape[1:]])
    optimum = tuple([int(options.get("opt_batch", 8)), *shape[1:]])
    maximum = tuple([int(options.get("max_batch", 32)), *shape[1:]])
    profile_result = profile.set_shape(input_tensor.name, minimum, optimum, maximum)
    # TensorRT 11 returns None on success; older releases return bool.
    if profile_result is False:
        raise RuntimeError(f"Invalid TensorRT batch profile: min={minimum}, opt={optimum}, max={maximum}")
    config.add_optimization_profile(profile)


def _configure_trt_verbosity(config, trt, options: dict[str, Any]) -> None:
    if not hasattr(config, "profiling_verbosity") or not hasattr(trt, "ProfilingVerbosity"):
        return
    key = str(options.get("profiling_verbosity", "layer_names_only")).lower()
    mapping = {
        "none": trt.ProfilingVerbosity.NONE,
        "layer_names_only": trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
        "detailed": trt.ProfilingVerbosity.DETAILED,
    }
    config.profiling_verbosity = mapping.get(key, trt.ProfilingVerbosity.LAYER_NAMES_ONLY)


def normalize_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"input_size must be an integer or [width, height], received {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Override export.checkpoint")
    parser.add_argument("--output-dir", type=Path, help="Override export.output_dir")
    parser.add_argument("--format", choices=("all", "onnx", "engine"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    export_cfg = cfg["export"]
    checkpoint = (args.checkpoint or Path(export_cfg["checkpoint"])).resolve()
    output_dir = (args.output_dir or Path(export_cfg["output_dir"])).resolve()
    onnx_path = output_dir / export_cfg["filenames"]["onnx"]
    engine_path = output_dir / export_cfg["filenames"]["engine"]
    targets = []
    if args.format in {"all", "onnx"} and bool(export_cfg.get("export_onnx", True)):
        targets.append("onnx")
    if args.format in {"all", "engine"} and bool(export_cfg.get("export_engine", True)):
        targets.append("engine")
    print(f"Checkpoint: {checkpoint}")
    print(f"Targets: {targets}")
    print(f"ONNX output: {onnx_path}")
    print(f"TensorRT output: {engine_path}")
    if args.dry_run:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print("Export configuration is valid.")
        return
    if not bool(export_cfg.get("enabled", True)):
        print("Export disabled by export.enabled=false")
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    device_value = export_cfg["common_args"].get("device", 0)
    device = torch.device(f"cuda:{device_value}" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(cfg, checkpoint, device)
    if "onnx" in targets or ("engine" in targets and not onnx_path.is_file()):
        onnx_path = export_onnx(cfg, model, checkpoint, onnx_path, device)
    if "engine" in targets:
        export_engine(cfg, onnx_path, engine_path)
    print("Export complete.")


if __name__ == "__main__":
    main()
