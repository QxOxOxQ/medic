from typing import Any

from qdrant_client import models

from rag.config import SETTINGS


def _build_quantization_config(config: dict[str, Any]) -> Any:
    if config["kind"] == "scalar":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=getattr(models.ScalarType, config["scalar_type"]),
                quantile=config["quantile"],
                always_ram=config["always_ram"],
            )
        )

    if config["kind"] == "binary":
        return models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                encoding=getattr(
                    models.BinaryQuantizationEncoding,
                    config["encoding"],
                ),
                always_ram=config["always_ram"],
            )
        )

    raise ValueError(f"Unknown quantization kind: {config['kind']}")


quantization_configs = {
    method_name: {
        "config": _build_quantization_config(config),
        "expected_speedup": config["expected_speedup"],
        "expected_compression": config["expected_compression"],
    }
    for method_name, config in SETTINGS["measurement"]["quantization_configs"].items()
}
