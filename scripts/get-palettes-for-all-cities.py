#!/usr/bin/env python                                                                                                                                                              
# -*- coding: utf-8 -*-                                                                                                                                                            

from __future__ import division, print_function

import colorsys
import glob
import os
import requests
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import skimage.color as color
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

__all__ = ["search"]

URL = "https://api.flickr.com/services/rest"
PHOTO_URL = "https://farm{farm}.staticflickr.com/{server}/{id}_{secret}_q.jpg"
LOCAL = "images"

OUTPUT_PATH = "palettes"
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

def search(query):
    base = os.path.join(LOCAL, query)
    
    if os.path.exists(base):
        print(base, "already exists. Skipping download...")
        
    else:
        params = dict(
            method="flickr.photos.search",
    #        api_key=os.environ["FLICKR_API_KEY"],
            api_key='b2d8f11842ff8171b54591dea5f43541',
            text=query,
            content_type=1,
            safe_search=1,
            sort="interestingness-desc",
            format="json",
            nojsoncallback=1,
        )

        r = requests.get(URL, params=params)
        r.raise_for_status()
        photos = r.json().get("photos", {}).get("photo", [])

        os.makedirs(base, exist_ok=True)

        print("Grabbing", query, "images from Flickr...")

        for i, photo in enumerate(photos):
            url = PHOTO_URL.format(**photo)
            r = requests.get(url)
            r.raise_for_status()

            im = "{0:03d}.jpg".format(i)
            fn = os.path.join(base, im)
            print(fn)
            with open(fn, "wb") as f:
                f.write(r.content)

def make_histogram(pattern=LOCAL+"/*/*.jpg", filename="hist.h5"):
    print("")
    print("Generating", filename, "file...")
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
#    plt.imshow(H[5], interpolation="nearest", cmap="viridis")
#    plt.savefig("slice.png")
    print("")

def main(city_name, pattern, n_colors=8, pixels_per_image=1024,
         hist_filename="hist.h5"):

    with h5py.File(hist_filename, "r") as f:
        bins = f["bins"][...]
        counts = f["counts"][...]
    weights = 1.0 / (1.0 + counts)

    ic = io.ImageCollection(pattern)

    print(city_name,": loading subset of pixels from each image...")
    # load a subset of the pixels from each image into a single array

    all_rgb = np.zeros((len(ic), pixels_per_image, 3))
    for i,im in enumerate(ic):
        norm_rgb = im[...,:3].astype(np.float64) / 255.
        rgb_flat = norm_rgb.reshape(-1, 3)
        all_rgb[i] = shuffle(rgb_flat)[:pixels_per_image]

    # flatten so we just have a long array of RGB values 

    all_rgb = all_rgb.reshape(-1, 3)

    x = all_rgb[:,0]
    y = all_rgb[:,1]
    z = all_rgb[:,2]

    # feature matrix

    print(city_name,": creating feature matrix...")
    
    X = np.vstack((x, y, z)).T

    inds = np.digitize(X, bins) - 1
    inds[inds < 0] = 0
    inds[inds >= len(bins) - 1] = len(bins) - 2
    probs = weights[inds[:, 0], inds[:, 1], inds[:, 2]]
    probs /= np.sum(probs)
    inds = np.random.choice(np.arange(len(probs)), size=len(probs), p=probs)
    X = X[inds]

    clf = KMeans(n_clusters=n_colors)
    clf.fit(X)
    centroids = clf.cluster_centers_

    lums = [colorsys.rgb_to_hsv(*rgb)[2] for rgb in centroids]
    rgb_clusters = centroids[np.argsort(lums)]
    
    fig,ax = plt.subplots(1,1,figsize=(16,1))
    ax.imshow(rgb_clusters.reshape(1,len(rgb_clusters),3), interpolation='nearest')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.savefig(os.path.join(OUTPUT_PATH, "{}.png".format(city_name)), bbox_inches="tight")

    np.savetxt(os.path.join(OUTPUT_PATH, "{}.csv".format(city_name)), rgb_clusters, delimiter=",")
    
    print(city_name,": color palette created!")
    print("")


if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object

    parser = ArgumentParser(description="")
    parser.add_argument("-c", "--cities", dest="city_names", required=True,
                        type=str, help="Name of file containing cities.")
    parser.add_argument("--ppi", dest="pixels_per_image", default=1024,
                        type=int, help="Number of pixels to grab from each image.")
    parser.add_argument("-n", "--n-colors", dest="n_colors", default=8,
                        type=int, help="Number of colors in the palette.")

    args = parser.parse_args()

    queries_file = open(args.city_names, "r")
    queries = list(queries_file.read().split("\n"))
    print("Searching for images of the following cities...")
    print(queries)
    print("")

    for q in queries:
        search(q)

    make_histogram()

    for q in queries:
        main(pattern=LOCAL+"/"+q+"/*.jpg", city_name=q,
             pixels_per_image=args.pixels_per_image, n_colors=args.n_colors)
