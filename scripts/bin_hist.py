#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import h5py
import numpy as np
import skimage.io as io
import matplotlib.pyplot as pl

__all__ = ["make_histogram"]


def make_histogram(pattern, filename="hist.h5"):
    ic = io.ImageCollection(pattern)
    samples = []
    for i, im in enumerate(ic):
        norm_rgb = im[..., :3].astype(np.float64) / 255.
        rgb_flat = norm_rgb.reshape(-1, 3)
        samples.append(rgb_flat)
    samples = np.concatenate(samples, axis=0)

    bins = np.linspace(0, 1, 16)
    H, _ = np.histogramdd(samples, (bins, bins, bins))
    H = H.astype(np.uint64)

    with h5py.File(filename, "w") as f:
        f.create_dataset("bins", data=bins)
        f.create_dataset("counts", data=H)

    pl.imshow(H[5], interpolation="nearest", cmap="viridis")
    pl.savefig("slice.png")


if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("pattern",
                        type=str, help="Glob pattern for image files to load.")
    args = parser.parse_args()

    make_histogram(args.pattern)
