#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import requests

__all__ = ["search"]

URL = "https://api.flickr.com/services/rest"
PHOTO_URL = "https://farm{farm}.staticflickr.com/{server}/{id}_{secret}_q.jpg"
LOCAL = "images"

def search(keywords):
    params = dict(
        method="flickr.photos.search",
        api_key=os.environ["FLICKR_API_KEY"],
        text=keywords,
        content_type=1,
        safe_search=1,
        sort="interestingness-desc",
        format="json",
        nojsoncallback=1,
    )

    r = requests.get(URL, params=params)
    r.raise_for_status()
    photos = r.json().get("photos", {}).get("photo", [])

    base = os.path.join(LOCAL, keywords)
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
    import sys
    search(" ".join(sys.argv[1:]))
