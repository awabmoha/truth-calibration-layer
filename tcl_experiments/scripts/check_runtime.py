from __future__ import annotations

import json
import platform

import torch
import transformers


def main() -> None:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["cuda_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["cuda_capability"] = f"{props.major}.{props.minor}"
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
