import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import triangulation as tri

# Constants
B = 12               # Distance between the cameras [cm]
f_pixel = 1066       # Camera lens focal length [pixels]
min_dep = 50
max_dep = 1500

# Initialize YOLO model
def initialize_yolo(model_path):
    return YOLO(model_path)

# Initialize feature detector and matcher
def initialize_feature_detector_and_matcher():
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher()
    return detector, matcher

# Initialize Matplotlib for real-time plotting
def initialize_plot():
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    trajectory, = ax.plot([], [], 'b-', label="Robot Path")  # Trajectory line
    ax.set_xlim(-700, 700)  # Adjust limits based on your scene
    ax.set_ylim(0, 1500)     # Adjust limits based on your scene
    ax.set_xlabel("X Position [cm]")
    ax.set_ylabel("Y Position [cm]")
    ax.legend()
    plt.title("Robot Trajectory in Real-World Coordinates")
    return fig, ax, trajectory

# Detect objects using YOLO
def detect_objects(frame, model):
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
    return boxes

# Draw bounding boxes on the frame
def draw_bounding_boxes(frame, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)

# Extract regions of interest (ROIs) within the bounding boxes
def extract_roi(frame, box):
    return frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

# Match features between two ROIs
def match_features(detector, matcher, roi_left, roi_right):
    kp1, des1 = detector.detectAndCompute(roi_left, None)
    kp2, des2 = detector.detectAndCompute(roi_right, None)
    if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return kp1, kp2, good_matches
    return None, None, []

# Calculate depth for matched points
def calculate_depth(pts1, pts2, box_left, box_right, B, f_pixel, min_dep, max_dep):
    depths = [tri.find_depth(pt2, pt1, B, f_pixel) for pt1, pt2 in zip(pts1, pts2) if min_dep < tri.find_depth(pt2, pt1, B, f_pixel) < max_dep]
    if len(depths) > 0:
        mean_depth = np.mean(depths)
        std_depth = np.std(depths)
        threshold = 2 * std_depth
        filtered_depths = [depth for depth in depths if abs(depth - mean_depth) < threshold]
        return np.mean(filtered_depths) if len(filtered_depths) > 0 else 0
    return 0

# Calculate real-world coordinates
def calculate_real_world_coordinates(box, depth_cv, half_width, height, f_pixel):
    box_center_x = (box[0] + box[2]) / 2
    box_center_y = (box[1] + box[3]) / 2
    real_x = (box_center_x - half_width / 2) * depth_cv / f_pixel
    real_y = box_center_y * depth_cv / f_pixel
    return real_x, real_y

# Update the trajectory plot
def update_plot(trajectory, x_positions, y_positions):
    trajectory.set_data(x_positions, y_positions)
    plt.draw()
    plt.pause(0.001)

# Visualize key points on the frames
def visualize_keypoints(frame, keypoints, color=(0, 255, 0), radius=5):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)

# Process a single frame
def process_frame(frame_left, frame_right, model, detector, matcher, half_width, height, trajectory, x_positions, y_positions):
    # Detect objects in both frames
    boxes_left = detect_objects(frame_left, model)
    boxes_right = detect_objects(frame_right, model)

    if len(boxes_left) == 0 or len(boxes_right) == 0:
        cv2.putText(frame_right, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_left, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame_left, frame_right

    # Draw bounding boxes
    draw_bounding_boxes(frame_left, boxes_left)
    draw_bounding_boxes(frame_right, boxes_right)

    # Extract ROIs
    roi_left = extract_roi(frame_left, boxes_left[0])
    roi_right = extract_roi(frame_right, boxes_right[0])

    # Match features
    kp1, kp2, good_matches = match_features(detector, matcher, roi_left, roi_right)

    if len(good_matches) > 0:
        # Adjust keypoints to the original frame coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        pts1[:, 0] += boxes_left[0][0]
        pts1[:, 1] += boxes_left[0][1]
        pts2[:, 0] += boxes_right[0][0]
        pts2[:, 1] += boxes_right[0][1]

        # Visualize key points on the frames
        visualize_keypoints(frame_left, [cv2.KeyPoint(x=pt[0], y=pt[1], size=10) for pt in pts1])
        visualize_keypoints(frame_right, [cv2.KeyPoint(x=pt[0], y=pt[1], size=10) for pt in pts2])

        # Calculate depth
        depth_cv = calculate_depth(pts1, pts2, boxes_left[0], boxes_right[0], B, f_pixel, min_dep, max_dep)

        if depth_cv > 0:
            # Calculate real-world coordinates
            real_x, real_y = calculate_real_world_coordinates(boxes_left[0], depth_cv, half_width, height, f_pixel)

            # Update trajectory
            x_positions.append(real_x)
            y_positions.append(real_y)
            update_plot(trajectory, x_positions, y_positions)

            print(f"Robot Position: X={real_x:.2f} cm, Y={real_y:.2f} cm, Depth={depth_cv:.2f} cm")

    return frame_left, frame_right

# Main function
def main():
    model = initialize_yolo('best.pt')
    detector, matcher = initialize_feature_detector_and_matcher()
    fig, ax, trajectory = initialize_plot()

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

        frame_left = frame[:, :half_width, :]
        frame_right = frame[:, half_width:, :]

        frame_left, frame_right = process_frame(frame_left, frame_right, model, detector, matcher, half_width, height, trajectory, x_positions, y_positions)

        combined_frame = np.hstack((frame_left, frame_right))
        scale_factor = 0.5
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        resized_frame = cv2.resize(combined_frame, new_dimensions)

        cv2.imshow("Stereo Frames with Bounding Boxes and Matched Points", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()