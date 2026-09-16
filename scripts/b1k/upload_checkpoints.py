#!/usr/bin/env python3
"""Publish queued B1K checkpoints without importing the trainer or CUDA."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusion_policy.b1k.checkpoint_upload import main

if __name__ == '__main__':
    raise SystemExit(main())
