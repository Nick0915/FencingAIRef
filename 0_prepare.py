#! /usr/bin/env python3

def prepare():
    import os

    if not os.path.exists('./Data'):
        os.mkdir('./Data')

    directories = [
        'YtDownloads',
        'YtDownloads/Sabre',
        'ScoreInfo',
        'ScoreInfo/Sabre',
        'Clips',
        'Clips/Sabre',
        'Clips/Sabre/Left',
        'Clips/Sabre/Right',
        'DownsampledClips',
        'DownsampledClips/Sabre',
        'DownsampledClips/Sabre/Left',
        'DownsampledClips/Sabre/Right',
        'OverlaidClips',
        'OverlaidClips/Sabre',
        'OverlaidClips/Sabre/Left',
        'OverlaidClips/Sabre/Right',
    ]
    for folder in directories:
        desired = f'./Data/{folder}'
        if not os.path.exists(desired):
            os.mkdir(desired)

if __name__ == "__main__":
    prepare()