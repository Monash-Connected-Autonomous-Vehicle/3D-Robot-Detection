import cv2

def create_hsv_mask(frame, lower_bound, upper_bound):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    return mask

def filter_keypoints_by_mask(keypoints, mask):
    filtered_keypoints = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        # Check if the keypoint is within the bounds of the mask
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:  
            if mask[y, x] == 255:  # Check if the keypoint is within the mask
                filtered_keypoints.append(kp)
    return filtered_keypoints


def visualize_mask_and_keypoints(frame, mask, keypoints):
    # Overlay the mask on the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Draw key points on the masked frame
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(masked_frame, (x, y), 5, (0, 255, 0), -1)  # Green circles for key points

    return masked_frame