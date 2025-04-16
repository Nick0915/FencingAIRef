#! python3

def prepare():
    import os

    if not os.path.exists('./Data'):
        os.mkdir('./Data')

    directories = ['YtDownloads', 'ScoreInfo/Sabre']
    for folder in directories:
        desired = f'./Data/{folder}'
        if not os.path.exists(desired):
            os.mkdir(desired)

if __name__ == "__main__":
    prepare()