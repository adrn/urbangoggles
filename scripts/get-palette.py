# Standard library
import os
import sys
import glob

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import skimage.color as color
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

def main(pattern, n_colors=8, pixels_per_image=1024, cluster_hsv=False):
    ic = io.ImageCollection(pattern)

    # load a subset of the pixels from each image into a single array
    all_rgb = np.zeros((len(ic), pixels_per_image, 3))
    for i,im in enumerate(ic):
        norm_rgb = im[...,:3].astype(np.float64) / 255.
        rgb_flat = norm_rgb.reshape(-1, 3)
        all_rgb[i] = shuffle(rgb_flat)[:pixels_per_image]

    # either cluster in hue-saturation-value space or in RGB
    if cluster_hsv:
        hsv_flat = color.rgb2hsv(all_rgb).reshape(-1,3)
        phi = 2*np.pi*hsv_flat[:,0]
        x = hsv_flat[:,1]*np.cos(phi)
        y = hsv_flat[:,1]*np.sin(phi)
        z = hsv_flat[:,2]
    else:
        # flatten so we just have a long array of RGB values
        all_rgb = all_rgb.reshape(-1, 3)

        x = all_rgb[:,0]
        y = all_rgb[:,1]
        z = all_rgb[:,2]

    # feature matrix
    X = np.vstack((x,y,z)).T

    clf = KMeans(n_clusters=n_colors)
    clf.fit(X)
    centroids = clf.cluster_centers_

    if cluster_hsv:
        rgb_clusters = color.hsv2rgb(centroids[None])[0]
    else:
        rgb_clusters = centroids

    fig,ax = plt.subplots(1,1,figsize=(16,1))
    ax.imshow(rgb_clusters.reshape(1,len(rgb_clusters),3), interpolation='nearest')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-f", "--files", dest="file_pattern", required=True,
                        type=str, help="Glob pattern for image files to load.")
    parser.add_argument("--hsv", dest="hsv", default=False, action="store_true",
                        help="Cluster in HSV instead of RGB.")
    parser.add_argument("--ppi", dest="pixels_per_image", default=1024,
                        type=int, help="Number of pixels to grab from each image.")

    args = parser.parse_args()

    main(args.file_pattern, pixels_per_image=args.pixels_per_image,
         cluster_hsv=args.hsv)
