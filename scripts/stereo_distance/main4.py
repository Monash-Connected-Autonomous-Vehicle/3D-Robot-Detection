import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import triangulation as tri
from HSV_filter import create_hsv_mask, filter_keypoints_by_mask, visualize_mask_and_keypoints

# Constants for stereo and depth calculation
B = 12               # Baseline: distance between the cameras [cm]
f_pixel = 1066       # Focal length in pixels
min_dep = 50
max_dep = 1500

# HSV bounds for light blue (adjust as necessary)
lower_blue = np.array([80, 50, 50])
upper_blue = np.array([130, 255, 255])

# Kalman Filter class (same as provided previously)
class KalmanFilter:
    def __init__(self, dt=1, process_noise=1e-5, measurement_noise=1e-1):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # State transition matrix (x, y, vx, vy, width, height, v_width, v_height)
        self.A = np.array([[1, 0, dt, 0, 0, 0, 0, 0],
                           [0, 1, 0, dt, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, dt, 0],
                           [0, 0, 0, 0, 0, 1, 0, dt],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

        # Measurement matrix (x, y, width, height)
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0]])

        self.Q = np.eye(8) * self.process_noise  # Process noise covariance
        self.R = np.eye(4) * self.measurement_noise  # Measurement noise covariance

        self.x = np.zeros((8, 1))  # Initial state: [x, y, vx, vy, width, height, v_width, v_height]
        self.P = np.eye(8)         # State covariance

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2]

    def update(self, z):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.x[:2]

# Initialization functions
def initialize_yolo(model_path):
    return YOLO(model_path)

def initialize_feature_detector_and_matcher():
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher()
    return detector, matcher

def initialize_plot():
    plt.ion()  # Enable interactive plotting
    fig, ax = plt.subplots()
    trajectory, = ax.plot([], [], 'b-', label="Robot Path")
    ax.set_xlim(-700, 700)
    ax.set_ylim(0, 1500)
    ax.set_xlabel("X Position [cm]")
    ax.set_ylabel("Y Position [cm]")
    ax.legend()
    plt.title("Robot Trajectory in Real-World Coordinates")
    return fig, ax, trajectory

# Object detection using YOLO
def detect_objects(frame, model):
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
    return boxes

def draw_bounding_boxes(frame, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)

def extract_roi(frame, box):
    return frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

def match_features(detector, matcher, roi_left, roi_right):
    # Detect keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(roi_left, None)
    kp2, des2 = detector.detectAndCompute(roi_right, None)
    if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
        matches = matcher.knnMatch(des1, des2, k=2)
        # Ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return kp1, kp2, good_matches
    return None, None, []

def calculate_depth(pts1, pts2, box_left, box_right, B, f_pixel, min_dep, max_dep):
    depths = [tri.find_depth(pt2, pt1, B, f_pixel) 
              for pt1, pt2 in zip(pts1, pts2)
              if min_dep < tri.find_depth(pt2, pt1, B, f_pixel) < max_dep]
    if len(depths) > 0:
        mean_depth = np.mean(depths)
        std_depth = np.std(depths)
        threshold = 2 * std_depth
        filtered_depths = [depth for depth in depths if abs(depth - mean_depth) < threshold]
        return np.mean(filtered_depths) if len(filtered_depths) > 0 else 0
    return 0

def calculate_real_world_coordinates(box, depth_cv, half_width, height, f_pixel):
    box_center_x = (box[0] + box[2]) / 2
    box_center_y = (box[1] + box[3]) / 2
    real_x = (box_center_x - half_width / 2) * depth_cv / f_pixel
    real_y = box_center_y * depth_cv / f_pixel
    return real_x, real_y

def update_plot(trajectory, x_positions, y_positions):
    trajectory.set_data(x_positions, y_positions)
    plt.draw()
    plt.pause(0.001)

def visualize_keypoints(frame, keypoints, color=(0, 255, 0), radius=5):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)

# Process a single frame pair using HSV filtering on the ROIs to restrict keypoints to light blue regions
def process_frame(frame_left, frame_right, model, detector, matcher, half_width, height, trajectory, x_positions, y_positions, kf):
    # Detect objects in both images using YOLO
    boxes_left = detect_objects(frame_left, model)
    boxes_right = detect_objects(frame_right, model)

    if len(boxes_left) == 0 or len(boxes_right) == 0:
        cv2.putText(frame_right, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_left, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame_left, frame_right

    # Draw bounding boxes on both images
    draw_bounding_boxes(frame_left, boxes_left)
    draw_bounding_boxes(frame_right, boxes_right)

    # Extract ROIs using the first detected bounding box from each image
    roi_left = extract_roi(frame_left, boxes_left[0])
    roi_right = extract_roi(frame_right, boxes_right[0])

    # Create an HSV mask for light blue
    mask_left = create_hsv_mask(roi_left, lower_blue, upper_blue)
    mask_right = create_hsv_mask(roi_right, lower_blue, upper_blue)

    # Use the masks directly in detectAndCompute to restrict keypoints to light blue regions
    kp1, des1 = detector.detectAndCompute(roi_left, mask_left)
    kp2, des2 = detector.detectAndCompute(roi_right, mask_right)

    # Match features using BFMatcher with the ratio test
    if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    else:
        good_matches = []

    if len(good_matches) > 0:
        # Adjust keypoint coordinates from ROI to full-frame coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        pts1[:, 0] += boxes_left[0][0]
        pts1[:, 1] += boxes_left[0][1]
        pts2[:, 0] += boxes_right[0][0]
        pts2[:, 1] += boxes_right[0][1]

        # Visualize keypoints on the full-frame images
        visualize_keypoints(frame_left, [cv2.KeyPoint(x=pt[0], y=pt[1], size=10) for pt in pts1])
        visualize_keypoints(frame_right, [cv2.KeyPoint(x=pt[0], y=pt[1], size=10) for pt in pts2])

        # Calculate depth using triangulation
        depth_cv = calculate_depth(pts1, pts2, boxes_left[0], boxes_right[0], B, f_pixel, min_dep, max_dep)
        if depth_cv > 0:
            # Convert bounding box center to real-world coordinates
            real_x, real_y = calculate_real_world_coordinates(boxes_left[0], depth_cv, half_width, height, f_pixel)
            # Get bounding box dimensions
            box_width = boxes_left[0][2] - boxes_left[0][0]
            box_height = boxes_left[0][3] - boxes_left[0][1]

            # Update the Kalman Filter with measurement [real_x, real_y, box_width, box_height]
            kf.predict()
            filtered = kf.update(np.array([[real_x], [real_y], [box_width], [box_height]]))

            # Update trajectory plot with filtered coordinates
            x_positions.append(filtered[0, 0])
            y_positions.append(filtered[1, 0])
            update_plot(trajectory, x_positions, y_positions)

            print(f"Robot Position: X={filtered[0, 0]:.2f} cm, Y={filtered[1, 0]:.2f} cm, Depth={depth_cv:.2f} cm")

    return frame_left, frame_right


def main():
    # Initialize YOLO model, feature detector/matcher, plot and Kalman filter
    model = initialize_yolo('best.pt')
    detector, matcher = initialize_feature_detector_and_matcher()
    fig, ax, trajectory = initialize_plot()
    kf = KalmanFilter(dt=1, process_noise=1e-5, measurement_noise=1e-1)

    video_path = "input.avi"
    cap = cv2.VideoCapture(video_path)

    x_positions = []
    y_positions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        half_width = width // 2

        # Split the frame into left and right stereo images
        frame_left = frame[:, :half_width, :]
        frame_right = frame[:, half_width:, :]

        frame_left, frame_right = process_frame(frame_left, frame_right, model, detector, matcher,
                                                  half_width, height, trajectory, x_positions, y_positions, kf)

        # Combine and resize frames for display
        combined_frame = np.hstack((frame_left, frame_right))
        scale_factor = 0.5
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        resized_frame = cv2.resize(combined_frame, new_dimensions)
        cv2.imshow("Stereo Frames with HSV Filtering and Tracking", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()