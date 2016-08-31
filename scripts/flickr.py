#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import requests

__all__ = ["search"]

URL = "https://api.flickr.com/services/rest"
PHOTO_URL = "https://farm{farm}.staticflickr.com/{server}/{id}_{secret}_q.jpg"
LOCAL = "images"

def search(query):
    params = dict(
        method="flickr.photos.search",
        api_key=os.environ["FLICKR_API_KEY"],
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

    base = os.path.join(LOCAL, query)
    os.makedirs(base, exist_ok=True)

    for i, photo in enumerate(photos):
        url = PHOTO_URL.format(**photo)
        r = requests.get(url)
        r.raise_for_status()

        fn = os.path.join(base, "{0:03d}.jpg".format(i))
        print(fn)
        with open(fn, "wb") as f:
            f.write(r.content)


if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-s", "--search", dest="search_query", required=True, type=str)

    args = parser.parse_args()

    search(args.search_query)
