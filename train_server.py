#!/usr/bin/env python3
"""Short-command launcher for server classifier training."""

from __future__ import annotations

import sys

from train_yolo_crop_classifier import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--config",
        "configs/convnextv2_nano_yolo_crops.yaml",
        "--environment",
        "server",
        *sys.argv[1:],
    ]
    main()
