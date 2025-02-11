from ultralytics import YOLO
import cv2
def load_model(model_path='best.pt'):
    """
    Load the YOLO model for pose detection.
    """
    return YOLO(model_path)

def load_video(video_path='input.avi'):
    """
    Load the video file for input.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    return cap

def initialize_video_writer(cap, output_path='output.avi'):
    """
    Initialize the video writer for saving the processed frames.
    """
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
    return out, frame_width, frame_height

def split_stereo_frame(frame, width):
    """
    Split the stereo frame into left and right frames.
    """
    left_frame = frame[:, :width // 2]  # Left half
    right_frame = frame[:, width // 2:]  # Right half
    return left_frame, right_frame

def process_frame_with_model(model, frame):
    """
    Process a frame with the YOLO model and return keypoints and bounding boxes.
    """
    results = model(frame)
    keypoints = results[0].keypoints.cpu().numpy()  # Extract keypoints
    boxes = results[0].boxes.xywh.cpu().numpy()  # Extract bounding boxes (x, y, w, h)
    return keypoints, boxes

def resize_frame(frame, scale_factor, width, height):
    """
    Resize the frame for output video size.
    """
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    return cv2.resize(frame, new_dimensions)

