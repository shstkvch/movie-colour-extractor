import cv2
import numpy as np
from sklearn.cluster import KMeans
import tkinter as tk
from multiprocessing import Pool, current_process

def extract_frame(video_path, timestamp):
    """ Extract a single frame from the video at the given timestamp. """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame = cap.read()
    cap.release()
    if success:
        print(f"Extracted frame at {timestamp} seconds by process {current_process().pid}")
        return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize frame to half the original size
    return None

def extract_frames(video_path, interval=60):
    """ Extract frames from the video at a given interval (in seconds). """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = frame_count / fps
    cap.release()

    timestamps = [i for i in range(0, int(video_length), interval)]
    pool = Pool()
    frames = pool.starmap(extract_frame, [(video_path, t) for t in timestamps])
    pool.close()
    pool.join()

    # Remove None values in case some frames failed to be read
    frames = [frame for frame in frames if frame is not None]
    return frames

def extract_colors(frames, k=5):
    """ Extract the dominant colors from the aggregated frames using K-means clustering. """
    all_pixels = np.vstack([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).reshape(-1, 3) for frame in frames])
    all_pixels = all_pixels[(all_pixels[:, 2] > 35) & (all_pixels[:, 2] < 220)]  # Filter out very dark and very bright colors
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(all_pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    dominant_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0] for color in dominant_colors]
    return [tuple(color) for color in dominant_colors]

def display_colors(colors):
    """ Display color swatches in a Tkinter window. """
    root = tk.Tk()
    root.title("Dominant Colors")
    for color in colors:
        color_hex = '#%02x%02x%02x' % color
        label = tk.Label(root, text=color_hex, bg=color_hex, fg='black' if sum(color) > 384 else 'white')
        label.pack(fill='both', expand=True)
    root.mainloop()

def process_video(video_path):
    frames = extract_frames(video_path)
    colors = extract_colors(frames)
    return colors

if __name__ == '__main__':
    # Example usage
    video_path = 'ts.mp4'
    colors = process_video(video_path)
    display_colors(colors)
