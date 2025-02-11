import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

def load_model(model_path='best.pt'):
    """Load the YOLO model for pose detection."""
    return YOLO(model_path)

def load_video(video_path='input.avi'):
    """Load the video file for input."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    return cap

def initialize_video_writer(cap, output_path='output.avi'):
    """Initialize the video writer for saving the processed frames."""
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
    return out, frame_width, frame_height

def split_stereo_frame(frame, width):
    """Split the stereo frame into left and right frames."""
    left_frame = frame[:, :width // 2]  # left half
    right_frame = frame[:, width // 2:]  # right half
    return left_frame, right_frame

def process_frame_with_model(model, frame):
    """
    Process a frame with the YOLO model and return keypoints.
    The returned keypoints are assumed to have shape (num_detections, num_keypoints, 3)
    where each keypoint is [x, y, confidence].
    """
    results = model(frame)
    # The following assumes that there is at least one detection.
    return results[0].keypoints.cpu().numpy()

def resize_frame(frame, scale_factor, width, height):
    """Resize the frame for output video size."""
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    return cv2.resize(frame, new_dimensions)

def draw_keypoints(frame, keypoints, scale_factor=0.5, conf_threshold=0.5):
    """Draw keypoints on a frame."""
    # keypoints is assumed to have a .data attribute if detections exist.
    if keypoints.data.size == 0:
        print("No keypoints detected.")
        return frame

    for person in keypoints.data:
        for kp in person:
            x, y, confidence = kp  # keypoint coordinates and confidence score
            if confidence > conf_threshold:
                cv2.circle(frame, (int(x * scale_factor), int(y * scale_factor)), 5, (0, 255, 0), -1)
    return frame

def triangulate_point_svd(P1, p1, P2, p2):
    """
    Triangulate a 3D point from two corresponding 2D points p1 and p2
    (each given as [x, y]) and the two 3x4 projection matrices P1 and P2.
    Returns the 3D point in non-homogeneous coordinates.
    """
    A = np.zeros((4, 4))
    # Each 2D point gives two equations. Here we form:
    #   p[0]*P[2,:] - P[0,:]  and  p[1]*P[2,:] - P[1,:]
    A[0] = p1[0] * P1[2] - P1[0]
    A[1] = p1[1] * P1[2] - P1[1]
    A[2] = p2[0] * P2[2] - P2[0]
    A[3] = p2[1] * P2[2] - P2[1]

    # Use SVD to solve A*X = 0
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]  # solution is the last row of V^T (or last column of V)
    X = X / X[3]  # normalize to make homogeneous coordinate 1
    return X[:3]

def estimate_robot_position(keypoints_left, keypoints_right, P_left, P_right, conf_threshold=0.5):
    """
    Given keypoints from the left and right images, estimate the 3D position of the robot.
    This example assumes that each detection returns exactly two keypoints (e.g., two diagonal corners).
    The robot's 3D position is taken as the midpoint between the triangulated 3D positions.
    """
    positions = []
    num_detections = keypoints_left.shape[0]
    for i in range(num_detections):
        # Extract the two keypoints for the current detection from each view.
        # Each keypoint is [x, y, confidence].
        try:
            kp1_left = keypoints_left[i, 0]
            kp2_left = keypoints_left[i, 1]
            kp1_right = keypoints_right[i, 0]
            kp2_right = keypoints_right[i, 1]

            # Only use keypoints with sufficient confidence.
            if (kp1_left[2] < conf_threshold or kp2_left[2] < conf_threshold or
                kp1_right[2] < conf_threshold or kp2_right[2] < conf_threshold):
                continue

            # Get the 2D coordinates.
            p1_left = kp1_left[:2]
            p2_left = kp2_left[:2]
            p1_right = kp1_right[:2]
            p2_right = kp2_right[:2]

            # Triangulate each corresponding pair.
            X1 = triangulate_point_svd(P_left, p1_left, P_right, p1_right)
            X2 = triangulate_point_svd(P_left, p2_left, P_right, p2_right)

            # Estimate the robot's 3D position as the midpoint of the two 3D points.
            robot_pos = (X1 + X2) / 2.0
            positions.append(robot_pos)
        except:
            pass
    return positions

def main():
    # Load the model and video
    model = load_model()
    cap = load_video()
    out, frame_width, frame_height = initialize_video_writer(cap)

    # Assume the stereo frame is arranged as left/right halves.
    # For triangulation, we need the projection matrices for the left and right cameras.
    # (In practice, these come from calibration.)
    # Here we assume a simple pinhole model.
    # Define some example intrinsic parameters.
    fx = fy = 1066
    cx = (frame_width // 2) / 2  # center of left frame (width is half of full frame)
    cy = frame_height / 2
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]])
    # For left camera: assume P_left = K [I | 0]
    P_left = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # For right camera: assume a baseline b (e.g., 0.1 meters) along the x-axis.
    b = 0.12
    T = np.array([[-b], [0], [0]])
    P_right = K @ np.hstack((np.eye(3), T))

    scale_factor = 0.5  # for display purposes

    # Enable interactive mode for Matplotlib
    plt.ion()
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], c='blue', marker='o')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Robot Position (X vs Y)')
    ax.grid(True)
    plt.show()

    # Initialize lists for storing coordinates
    x_coords = []
    y_coords = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Split the stereo frame into left and right images.
        left_frame, right_frame = split_stereo_frame(frame, frame_width)

        # Process both frames with the model.
        keypoints_left = process_frame_with_model(model, left_frame)
        keypoints_right = process_frame_with_model(model, right_frame)

        # (Optional) Resize frames for visualization.
        resized_left = resize_frame(left_frame, scale_factor, frame_width // 2, frame_height)
        resized_right = resize_frame(right_frame, scale_factor, frame_width // 2, frame_height)

        # Draw keypoints for visualization.
        resized_left = draw_keypoints(resized_left, keypoints_left, scale_factor)
        resized_right = draw_keypoints(resized_right, keypoints_right, scale_factor)

        # Combine the frames side-by-side.
        combined_frame = np.hstack((resized_left, resized_right))

        # If detections are available, estimate the 3D robot position.
        if keypoints_left.data.size and keypoints_right.data.size:
            positions = estimate_robot_position(keypoints_left.data, keypoints_right.data, P_left, P_right)

            for pos in positions:
                x, y, z = pos[0], pos[1], pos[2]
                x_coords.append(x)
                y_coords.append(y)

                # Create a text string for visualization on the frame
                text = f"Robot 3D pos: X={x:.2f}, Y={y:.2f}, Z={z:.2f}"
                cv2.putText(combined_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                print(text)

            # Update the scatter plot dynamically
            sc.set_offsets(np.c_[x_coords, y_coords])
            ax.set_xlim(-5, 5)  # Set X-axis limits
            ax.set_ylim(-5, 5)  # Set Y-axis limits
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)  # Small pause to update the plot

        # Show the combined frame with the overlayed text.
        cv2.imshow('Stereo Pose Visualization', combined_frame)
        out.write(combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
