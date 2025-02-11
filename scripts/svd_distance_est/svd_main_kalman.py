import cv2
import numpy as np
from util import *



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

def initialize_kalman_filter():
    """
    Initialize the Kalman filter.
    """
    kf = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)

    # State transition matrix (A)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

    # Measurement matrix (H)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

    # Process noise covariance matrix (Q)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

    # Measurement noise covariance matrix (R)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

    # Initial state estimate (x)
    kf.statePost = np.zeros((4, 1), np.float32)

    return kf

def update_kalman_filter(kf, measurement):
    """
    Update the Kalman filter with a new measurement.
    """
    # Predict the next state
    prediction = kf.predict()

    # Update the state with the new measurement
    kf.correct(measurement)

    return prediction

def predict_keypoints_from_bbox(prev_bbox, curr_bbox, prev_keypoints):
    """
    Predict keypoint positions based on changes in the bounding box.
    """
    if prev_bbox is None or curr_bbox is None:
        return prev_keypoints

    # Calculate changes in bounding box position and size
    dx = curr_bbox[0] - prev_bbox[0]  # Change in x
    dy = curr_bbox[1] - prev_bbox[1]  # Change in y
    dw = curr_bbox[2] / prev_bbox[2]  # Change in width (scale factor)
    dh = curr_bbox[3] / prev_bbox[3]  # Change in height (scale factor)

    # Predict keypoint positions
    predicted_keypoints = prev_keypoints.copy()
    for person in predicted_keypoints:
        for kp in person:
            kp[0] = (kp[0] - prev_bbox[0]) * dw 
            #+ curr_bbox[0] + dx  # Adjust x
            kp[1] = (kp[1] - prev_bbox[1]) * dh 
            #+ curr_bbox[1] + dy  # Adjust y

    return predicted_keypoints

def main():
    # Load the model and video
    model = load_model()
    cap = load_video()
    out, frame_width, frame_height = initialize_video_writer(cap)

    # Initialize the Kalman filter
    kf = initialize_kalman_filter()

    # Variables to store previous bounding box and keypoints
    prev_bbox_left = None
    prev_bbox_right = None
    prev_keypoints_left = None
    prev_keypoints_right = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Split the stereo frame into left and right
        left_frame, right_frame = split_stereo_frame(frame, frame_width)

        # Process both frames
        keypoints_left_raw, bbox_left= process_frame_with_model(model, left_frame)
        keypoints_right_raw, bbox_right = process_frame_with_model(model, right_frame)

        # Convert pose detection from KeyPoints Data Structure to Keypoints
        keypoints_left = keypoints_left_raw.data
        keypoints_right = keypoints_right_raw.data

        # Predict keypoints based on bounding box changes
        if prev_bbox_left is not None and bbox_left.size > 0:
            keypoints_left = predict_keypoints_from_bbox(prev_bbox_left, bbox_left[0], prev_keypoints_left)
        if prev_bbox_right is not None and bbox_right.size > 0:
            keypoints_right = predict_keypoints_from_bbox(prev_bbox_right, bbox_right[0], prev_keypoints_right)

        # Update previous bounding box and keypoints
        prev_bbox_left = bbox_left[0] if bbox_left.size > 0 else None
        prev_bbox_right = bbox_right[0] if bbox_right.size > 0 else None
        prev_keypoints_left = keypoints_left
        prev_keypoints_right = keypoints_right

        # Resize frames
        scale_factor = 0.5
        resized_left = resize_frame(left_frame, scale_factor, frame_width // 2, frame_height)
        resized_right = resize_frame(right_frame, scale_factor, frame_width // 2, frame_height)

        # Draw keypoints
        #resized_left = draw_keypoints(resized_left, keypoints_left, scale_factor)
        #resized_right = draw_keypoints(resized_right, keypoints_right, scale_factor)

        # Combine the left and right frames for visualization
        combined_frame = np.hstack((resized_left, resized_right))

        # Update the Kalman filter with the detected keypoints
        if keypoints_left.size > 0:
            for person in keypoints_left:
                for kp in person:
                    x, y, confidence = kp
                    if confidence > 0.5:
                        measurement = np.array([[x], [y]], np.float32)
                        prediction = update_kalman_filter(kf, measurement)
                        # Draw the predicted position
                        cv2.circle(combined_frame, (int(prediction[0] * scale_factor), int(prediction[1] * scale_factor)), 5, (255, 0, 0), -1)

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