#! python3

# ! Make sure `0_prepare.py` is run first!

SABRE_OUTPUT_DIR = './Data/YtDownloads/Sabre/'
MAX_VIDS = 25

# allows us downloading YT videos to local
from pytubefix import YouTube as YT

def download_vid(vid_id: int, link: str):
    video = YT(link)

    # example: 0024sabre.mp4
    filename = f'{vid_id:04}sabre.mp4'

    # mp4 streams only
    # DASH streams get messed up on download, so use progressive only
    # take only 360p streams
    video\
        .streams\
        .filter(progressive=True, file_extension='mp4', resolution='360p')\
        .first()\
        .download(
            output_path=SABRE_OUTPUT_DIR,
            filename=filename,
            skip_existing=True,
            timeout=5
        )
    print(f'Finished downloading {vid_id}: {link}', flush=True)

    del video

# [download_vid(i, l) for i, l in enumerate(sabre_links[:MAX_VIDS])]

import multiprocessing as mp
import numpy as np

def get_videos(links, num: int = None):
    if num is None:
        num = len(sabre_links)

    tasks = [(vid_id, link) for vid_id, link in enumerate(links[:num])]

    with mp.Pool(processes=25) as pool:
        pool.starmap(download_vid, tasks)

if __name__ == '__main__':
    # read all links from txt file
    with open('./sabre_vids.txt', 'r') as links_file:
        sabre_links = [link.strip() for link in links_file.readlines()]
        get_videos(sabre_links, MAX_VIDS)
