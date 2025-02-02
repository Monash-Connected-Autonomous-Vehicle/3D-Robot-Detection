import numpy as np
import cv2

# Filtering kernel
kernel = np.ones((3, 3), np.uint8)

def coords_mouse_disp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        average = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disp[y + u, x + v]
        average = average / 9
        Distance = -593.97 * average**3 + 1506.8 * average**2 - 1373.1 * average + 522.06
        Distance = np.around(Distance * 0.01, decimals=2)
        print('Distance:', Distance, 'm')

# Load stereo video
video = cv2.VideoCapture('input.avi')

# Check if the video opened successfully
if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get frame width and height
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Assume the left and right images are side by side in the video
half_width = frame_width // 2

# Stereo matching parameters
window_size = 3
min_disp = 2
num_disp = 130 - min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=5,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2
)

# WLS FILTER
lmbda = 80000
sigma = 1.8
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
stereoR = cv2.ximgproc.createRightMatcher(stereo)

while True:
    ret, frame = video.read()
    if not ret:
        break  # Stop if video ends

    # Split left and right images
    frameL = frame[:, :half_width]  # Left image
    frameR = frame[:, half_width:]  # Right image

    # Convert to grayscale
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Compute disparity
    disp = stereo.compute(grayL, grayR)
    dispL = np.int16(disp)
    dispR = np.int16(stereoR.compute(grayR, grayL))

    # Apply WLS filter
    filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
    filteredImg = cv2.normalize(filteredImg, filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    # Apply colormap
    filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

    # Show result
    cv2.imshow('Filtered Color Depth', filt_Color)
    cv2.setMouseCallback("Filtered Color Depth", coords_mouse_disp, filt_Color)

    # Exit when space bar is pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release resources
video.release()
cv2.destroyAllWindows()