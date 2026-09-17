#!/usr/bin/env python3
"""Build and verify the pixel-exact resized-frame cache used by `train_b1k.py --frame-cache`."""

from diffusion_policy.b1k.frame_cache import main

if __name__ == '__main__':
    main()
