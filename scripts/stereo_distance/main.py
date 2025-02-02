import cv2
import numpy as np
from ultralytics import YOLO
import triangulation as tri

# Load YOLO model
model = YOLO('best.pt')

# Open the video file
video_path = "input.avi"
cap = cv2.VideoCapture(video_path)

B = 12               # Distance between the cameras [cm]
f_pixel = 1066       # Camera lens focal length [pixels]
min_dep = 50
max_dep = 1500 
# Initialize feature detector and matcher
detector = cv2.SIFT_create()
matcher = cv2.BFMatcher()

while cap.isOpened():
    ret, frame = cap.read()

    # If the frame cannot be read, break the loop
    if not ret:
        break

    # Split the frame into left and right halves
    height, width, _ = frame.shape
    half_width = width // 2

    frame_left = frame[:, :half_width, :]
    frame_right = frame[:, half_width:, :]

    # Run YOLO model on both frames
    results_left = model(frame_left)
    results_right = model(frame_right)

    # Extract bounding boxes from YOLO results
    boxes_left = results_left[0].boxes.xyxy.cpu().numpy() if len(results_left[0].boxes) > 0 else []
    boxes_right = results_right[0].boxes.xyxy.cpu().numpy() if len(results_right[0].boxes) > 0 else []

    # If no objects are detected in one of the views, show "TRACKING LOST"
    if len(boxes_left) == 0 or len(boxes_right) == 0:
        cv2.putText(frame_right, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_left, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        # Assuming one object per frame for simplicity
        box_left = boxes_left[0]
        box_right = boxes_right[0]

        # Draw bounding boxes on the frames
        cv2.rectangle(frame_left, (int(box_left[0]), int(box_left[1])), (int(box_left[2]), int(box_left[3])), (0, 255, 0), 2)
        cv2.rectangle(frame_right, (int(box_right[0]), int(box_right[1])), (int(box_right[2]), int(box_right[3])), (0, 255, 0), 2)

        # Extract regions of interest (ROIs) within the bounding boxes
        roi_left = frame_left[int(box_left[1]):int(box_left[3]), int(box_left[0]):int(box_left[2])]
        roi_right = frame_right[int(box_right[1]):int(box_right[3]), int(box_right[0]):int(box_right[2])]

        # Detect features and compute descriptors within the ROIs
        kp1, des1 = detector.detectAndCompute(roi_left, None)
        kp2, des2 = detector.detectAndCompute(roi_right, None)

        if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
            # Match features using the BFMatcher
            matches = matcher.knnMatch(des1, des2, k=2)

            # Apply ratio test to filter good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # Calculate depth using traditional CV method
            if len(good_matches) > 0:
                # Adjust keypoints to the original frame coordinates
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

                # Adjust for ROI offset
                pts1[:, 0] += box_left[0]
                pts1[:, 1] += box_left[1]
                pts2[:, 0] += box_right[0]
                pts2[:, 1] += box_right[1]

                # Calculate depth for each matched point
                depths = []
                for pt1, pt2 in zip(pts1, pts2):
                    depth = tri.find_depth(pt2, pt1, B, f_pixel)
                    if depth > min_dep and depth < max_dep:
                        depths.append(depth)

                # Statistical Filtering: 
                # Calculate mean and standard deviation of depths
                if len(depths) > 0:
                    mean_depth = np.mean(depths)
                    std_depth = np.std(depths)

                    # Define a threshold (e.g., 2 standard deviations from the mean)
                    threshold = 2 * std_depth

                    # Filter out outliers
                    filtered_depths = [depth for depth in depths if abs(depth - mean_depth) < threshold]

                    # Use the filtered depths
                    if len(filtered_depths) > 0:
                        depth_cv = np.mean(filtered_depths)
                    else:
                        depth_cv = 0  # No valid depths after filtering
                else:
                    depth_cv = 0  # No valid depths available
                    
                # Visualize matched points on the frames
                for pt1, pt2 in zip(pts1, pts2):
                    cv2.circle(frame_left, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 0), -1)
                    cv2.circle(frame_right, (int(pt2[0]), int(pt2[1])), 5, (255, 0, 0), -1)

                # Display tracking and depth information
                cv2.putText(frame_right, "TRACKING", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
                cv2.putText(frame_left, "TRACKING", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
                cv2.putText(frame_right, f"Distance: {round(depth_cv, 3)}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
                cv2.putText(frame_left, f"Distance: {round(depth_cv, 3)}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)

                print("Depth: ", depth_cv)
            else:
                depth_cv = 0
        else:
            depth_cv = 0

    # Combine left and right frames side by side
    combined_frame = np.hstack((frame_left, frame_right))

    # Resize the combined frame to a smaller size
    scale_factor = 0.5  # Adjust this value to scale the frame (e.g., 0.5 for half-size)
    height, width, _ = combined_frame.shape
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    resized_frame = cv2.resize(combined_frame, new_dimensions)

    # Show the resized combined frame
    cv2.imshow("Stereo Frames with Bounding Boxes and Matched Points", resized_frame)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()