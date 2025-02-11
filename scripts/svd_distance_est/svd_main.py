import cv2
import numpy as np
from ultralytics import YOLO

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
    Process a frame with the YOLO model and return keypoints.
    """
    results = model(frame)
    return results[0].keypoints.cpu().numpy()  # Extract keypoints

def resize_frame(frame, scale_factor, width, height):
    """
    Resize the frame for output video size.
    """
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    return cv2.resize(frame, new_dimensions)

def draw_keypoints(frame, keypoints, scale_factor=0.5):
    """
    Draw keypoints on a frame.
    """
    if keypoints.data.size == 0:  # Check if keypoints are empty
        print("No keypoints detected.")
        return frame

    for person in keypoints.data:
        for kp in person:
            x, y, confidence = kp  # Keypoint coordinates and confidence score
            if confidence > 0.5:  # Only draw if confidence is above a threshold
                cv2.circle(frame, (int(x * scale_factor), int(y * scale_factor)), 5, (0, 255, 0), -1)
    return frame

def main():
    # Load the model and video
    model = load_model()
    cap = load_video()
    out, frame_width, frame_height = initialize_video_writer(cap)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Split the stereo frame into left and right
        left_frame, right_frame = split_stereo_frame(frame, frame_width)

        # Process both frames
        keypoints_left = process_frame_with_model(model, left_frame)
        keypoints_right = process_frame_with_model(model, right_frame)

        # Resize frames
        scale_factor = 0.5
        resized_left = resize_frame(left_frame, scale_factor, frame_width // 2, frame_height)
        resized_right = resize_frame(right_frame, scale_factor, frame_width // 2, frame_height)

        # Draw keypoints
        resized_left = draw_keypoints(resized_left, keypoints_left, scale_factor)
        resized_right = draw_keypoints(resized_right, keypoints_right, scale_factor)

        # Combine the left and right frames for visualization
        combined_frame = np.hstack((resized_left, resized_right))

        # Show the combined frame
        cv2.imshow('Stereo Pose Visualization', combined_frame)

        # Write the combined frame to the output video
        out.write(combined_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
