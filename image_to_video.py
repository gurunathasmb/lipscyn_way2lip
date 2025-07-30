import cv2

def image_to_video(image_path, output_video='face_video1.mp4', fps=25, duration=5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    h, w = img.shape[:2]
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    except AttributeError:
        raise ImportError("cv2 does not have VideoWriter_fourcc. Please ensure you have the correct version of OpenCV installed.")
    video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    if not video.isOpened():
        raise IOError(f"Could not open video writer for file: {output_video}")
    for _ in range(int(fps * duration)):
        video.write(img)
    video.release()
    return output_video