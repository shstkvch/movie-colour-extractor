import cv2
import numpy as np
from sklearn.cluster import KMeans
import tkinter as tk
from multiprocessing import Pool
import colorsys

def extract_frame(video_path, timestamp):
    """ Extract a single frame from the video at the given timestamp. """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame = cap.read()
    cap.release()
    return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) if success else None

def extract_frames(video_path, num_frames=64):
    """ Extract `num_frames` evenly spaced frames from the video. """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = frame_count / fps
    cap.release()

    timestamps = np.linspace(0, video_length, num_frames, endpoint=False)
    with Pool() as pool:
        frames = pool.starmap(extract_frame, [(video_path, t) for t in timestamps])

    return [frame for frame in frames if frame is not None]

def extract_colors(frames, k=32):
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

def adjust_color_hsv(h, s, v, s_factor=1.1, v_factor=1.1):
    """Increase the color saturation and brightness by a given factor."""
    s = min(s * s_factor, 1.0)
    v = min(v * v_factor, 1.0)
    return (h, s, v)

def create_complementary_color(hsv):
    """Create a complementary color."""
    h, s, v = hsv
    h = (h + 0.5) % 1.0  # Add 180 degrees in HSV
    return (h, s, v)

def create_aesthetic_palette(extracted_colors, num_colors=5):
    """Create an aesthetic palette from extracted colors."""
    # Convert RGB to HSV for better color manipulation
    hsv_colors = [colorsys.rgb_to_hsv(*color) for color in extracted_colors]

    # Sort colors by value and pick the one in the middle as base color
    base_color_hsv = sorted(hsv_colors, key=lambda x: x[2])[len(hsv_colors) // 2]

    # Create a harmonious palette
    palette_hsv = [adjust_color_hsv(*base_color_hsv)]
    while len(palette_hsv) < num_colors:
        # Create complementary colors
        new_color_hsv = create_complementary_color(palette_hsv[-1])
        palette_hsv.append(new_color_hsv)

    # Convert HSV back to RGB
    palette_rgb = [colorsys.hsv_to_rgb(*color) for color in palette_hsv]
    palette_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in palette_rgb]
    
    return palette_rgb

def process_video(video_path):
    frames = extract_frames(video_path)
    extracted_colors = extract_colors(frames)
    return extracted_colors

if __name__ == '__main__':
    # Example usage
    video_path = 'bb.mp4'
    colors = process_video(video_path)
    display_colors(colors)
