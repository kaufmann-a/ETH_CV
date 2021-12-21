import cv2
import os
import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    cutted_frame = frame[ymin:ymax, xmin:xmax, :]
    ch1 = np.histogram(cutted_frame[:, :, 0], hist_bin, range=(0, 256))
    ch2 = np.histogram(cutted_frame[:, :, 1], hist_bin, range=(0, 256))
    ch3 = np.histogram(cutted_frame[:, :, 2], hist_bin, range=(0, 256))
    hist = [ch1[0], ch2[0], ch3[0]]
    hist = hist / np.sum(hist)
    return hist


if __name__ == "__main__":
    video_name = "video2.avi"
    data_dir = './data/'
    video_path = os.path.join(data_dir, video_name)
    first_frame = 1
    last_frame = 60
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(1, first_frame)
    ret, first_image = vidcap.read()
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    hist = color_histogram(30, 30, 60, 60, first_image, 16)
    print(hist)