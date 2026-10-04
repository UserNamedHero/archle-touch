import cv2
import time
import mediapipe as mp

MARGIN = 0.15
# Active rectangle, in normalized (0-1) frame coordinates
X_LOW, X_HIGH = MARGIN, 1 - MARGIN
Y_LOW, Y_HIGH = MARGIN, 1 - MARGIN

# Inset box that stands in for the screen (pixels, top-right corner)
INSET_W, INSET_H = 160, 120

h, w, _ = (480, 640, 3)  # Default resolution for the camera feed
INSET_X, INSET_Y = w - INSET_W - 10, 10

cap = cv2.VideoCapture(0)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO, num_hands=2
)

# Create a larger default window for the camera feed.
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 2*w, 2*h)

hand_landmarker = HandLandmarker.create_from_options(options)

def remap(value, low, high):
    # Perform the linear interpolation
    remapped = (value - low) / (high - low)

    # Clamp the result between 0.0 and 1.0
    return max(0.0, min(remapped, 1.0))

while True:
    # Previous time
    previous_time = time.time()

    # Read the Camera input.
    ok, frame = cap.read()

    if not ok:
        break

    # Flip the frame horizontally for a selfie-view display.
    frame = cv2.flip(frame, 1)

    timestamp_ms = int(round(time.time() * 1000))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=rgb
    )

    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    # Draw the active rectangle every frame (it doesn't depend on the hand).
    # Corners are converted from normalized coords to pixels.
    cv2.rectangle(
        frame,
        (int(X_LOW * w), int(Y_LOW * h)),
        (int(X_HIGH * w), int(Y_HIGH * h)),
        (255, 255, 0), 2,
    )

    # Minimap rectangle (inset box) is drawn every frame (it doesn't depend on the hand).
    cv2.rectangle(
        frame,
        (INSET_X, INSET_Y),
        (INSET_X + INSET_W, INSET_Y + INSET_H),
        (255, 255, 255), 2,
    )

    if result.hand_landmarks:
        # Loop through all detected hands (up to 2)
        for hand_landmarks in result.hand_landmarks:
            # Index 4 is the Thumb Tip
            thumb_tip = hand_landmarks[4]
            # Index 8 is the Index Finger Tip
            index_finger_tip = hand_landmarks[8]

            # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
            index_finger_x = int(index_finger_tip.x * w)
            index_finger_y = int(index_finger_tip.y * h)
            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            # Remap using the NORMALIZED values, since the limits are normalized too.
            rx = remap(index_finger_tip.x, X_LOW, X_HIGH)
            ry = remap(index_finger_tip.y, Y_LOW, Y_HIGH)

            # Place the matching dot inside the inset box.
            dot_x = int(INSET_X + rx * INSET_W)
            dot_y = int(INSET_Y + ry * INSET_H)
            cv2.circle(frame, (dot_x, dot_y), 5, (0, 255, 255), -1)

            # Draw a green dot on the index finger tip
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 255, 0), -1)
            # Draw a red dot on the thumb tip
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), -1)

    # FPS calculation and Display
    elapsed_time = time.time() - previous_time
    if elapsed_time > 0:
        fps_text = str(round(1/elapsed_time, 2)) + " fps"
    else:
        fps_text = "0 fps"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame after modification.
    cv2.imshow("Camera", frame)

    # Wait for the 'q' key to exit the loop.
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()