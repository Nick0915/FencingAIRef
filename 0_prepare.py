#! python3

import os

if not os.path.exists('./Data'):
    os.mkdir('./Data')

directories = ['YtDownloads', '']
for folder in directories:
    desired = f'./Data/{folder}'
    if not os.path.exists(desired):
        os.mkdir(desired)