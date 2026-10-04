import cv2
import time
import mediapipe as mp

WIDTH = 640
HEIGHT = 480

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
cv2.resizeWindow("Camera", 2*WIDTH, 2*HEIGHT)
h, w, _ = (480, 640, 3)  # Default resolution for the camera feed
hand_landmarker = HandLandmarker.create_from_options(options)

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
            
            # Draw a green dot on the index finger tip
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 255, 0), -1)
            # Draw a red dot on the thumb tip
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), -1)

    # FPS calculation and Display
    elapsed_time = time.time() - previous_time
    if elapsed_time > 0:
        fps_text = str(round(1/elapsed_time, 2)) + " fps" # 1000 ms = 1 s
    else:
        fps_text = "0 fps"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show the frame after modification.
    cv2.imshow("Camera", frame)

    # Wait for the 'q' key to exit the loop.
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Safely print shape if a frame exists
if 'frame' in locals():
    print(frame.shape)  

hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()