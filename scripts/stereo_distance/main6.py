import cv2
import numpy as np

# HSV bounds for light blue
lower_blue = np.array([85, 50, 50])
upper_blue = np.array([160, 255, 255])

# Scale factor for resizing windows
scale_factor = 0.5  # Adjust this value to make the windows smaller or larger

def create_hsv_mask(frame, lower_bound, upper_bound):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    return mask

def visualize_mask_and_keypoints(frame, mask, keypoints):
    # Overlay the mask on the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Draw key points on the masked frame
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(masked_frame, (x, y), 5, (0, 255, 0), -1)  # Green circles for key points

    return masked_frame

def main():
    video_path = "input.avi"
    cap = cv2.VideoCapture(video_path)

    # Create resizable windows
    cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HSV Filtering and Keypoints Visualization", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        half_width = width // 2

        # Split the frame into left and right stereo images
        frame_left = frame[:, :half_width, :]
        frame_right = frame[:, half_width:, :]

        # Create an HSV mask for light blue for each frame
        mask_left = create_hsv_mask(frame_left, lower_blue, upper_blue)
        mask_right = create_hsv_mask(frame_right, lower_blue, upper_blue)

        # Detect keypoints in the left and right frames using the masks
        detector = cv2.SIFT_create()
        kp1, _ = detector.detectAndCompute(frame_left, mask_left)
        kp2, _ = detector.detectAndCompute(frame_right, mask_right)

        # Visualize the masks and keypoints
        masked_frame_left = visualize_mask_and_keypoints(frame_left, mask_left, kp1)
        masked_frame_right = visualize_mask_and_keypoints(frame_right, mask_right, kp2)

        # Combine the HSV-filtered frames
        hsv_filtered_frame = np.hstack((masked_frame_left, masked_frame_right))

        # Resize the frames for smaller windows
        resized_original = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
        resized_hsv_filtered = cv2.resize(hsv_filtered_frame, (int(width * scale_factor), int(height * scale_factor)))

        # Display the original video
        cv2.imshow("Original Video", resized_original)

        # Display the HSV-filtered video with keypoints
        cv2.imshow("HSV Filtering and Keypoints Visualization", resized_hsv_filtered)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()