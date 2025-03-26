import cv2 as cv

def vid_to_frames(filepath):
    cap = cv.VideoCapture(filepath)

    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break

        frames.append(frame)

    return frames