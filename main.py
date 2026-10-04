from collections import deque
import statistics
import cv2
import time
import mediapipe as mp
import math
from evdev import UInput, AbsInfo, ecodes as e
 
MARGIN = 0.15
# Active rectangle, in normalized (0-1) frame coordinates
X_LOW, X_HIGH = MARGIN, 1 - MARGIN
Y_LOW, Y_HIGH = MARGIN, 1 - MARGIN
 
# Inset box that stands in for the screen (pixels, top-right corner)
INSET_W, INSET_H = 160, 120
 
h, w, _ = (480, 640, 3)  # Default resolution for the camera feed
INSET_X, INSET_Y = w - INSET_W - 10, 10
 
# Which landmark moves the pointer. 8 = index fingertip.
# If the cursor jumps when you pinch, try 5 (index knuckle): it moves less during a pinch.
POINTER_LANDMARK = 8
 
# ---- virtual mouse (from exercise 6) ----
ABS_MAX = 32767
 
capabilities = {
    e.EV_KEY: [e.BTN_LEFT],
    e.EV_ABS: [
        (e.ABS_X, AbsInfo(value=0, min=0, max=ABS_MAX, fuzz=0, flat=0, resolution=0)),
        (e.ABS_Y, AbsInfo(value=0, min=0, max=ABS_MAX, fuzz=0, flat=0, resolution=0)),
    ],
}
 
ui = UInput(capabilities, name="hand-mouse")
time.sleep(1)  # let the desktop notice the new device
 
 
def move(nx, ny):
    ui.write(e.EV_ABS, e.ABS_X, int(nx * ABS_MAX))
    ui.write(e.EV_ABS, e.ABS_Y, int(ny * ABS_MAX))
    ui.syn()
 
 
def click(down):
    ui.write(e.EV_KEY, e.BTN_LEFT, int(down))
    ui.syn()
 
 
# ---- camera + hand tracking ----
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
    remapped = (value - low) / (high - low)
 
    # Clamp the result between 0.0 and 1.0
    return max(0.0, min(remapped, 1.0))
 
 
def smooth(prev, new, alpha):
    if prev is None:
        return new
    return prev + alpha * (new - prev)
 
 
def to_px(lm, w, h):
    return (lm.x * w, lm.y * h)
 
 
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
 
 
def pinch_ratio(hand, w, h):
    thumb = to_px(hand[4], w, h)
    index = to_px(hand[8], w, h)
    wrist = to_px(hand[0], w, h)
    knuckle = to_px(hand[5], w, h)  # landmark 5 = index knuckle
    size = dist(wrist, knuckle)
    return dist(thumb, index) / max(size, 1e-6)  # Avoid division by zero
 
 
def update_pinch(pinched, ratio, on, off):
    if not pinched:
        if ratio < on:
            return True
    else:
        if ratio > off:
            return False
    return pinched
 
 
# before the loop
sx = None
sy = None
ALPHA = 0.35
raw_history = deque(maxlen=30)
smooth_history = deque(maxlen=30)
 
pinched = False
PINCH_ON = 0.20
PINCH_OFF = 0.50
 
# Mouse control starts OFF so the cursor doesn't run away at launch.
# Press "c" in the camera window to toggle it.
control_enabled = False
button_down = False
 
try:
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
 
        # Active rectangle (fixed, same every frame)
        cv2.rectangle(
            frame,
            (int(X_LOW * w), int(Y_LOW * h)),
            (int(X_HIGH * w), int(Y_HIGH * h)),
            (255, 255, 0), 2,
        )
 
        # Minimap rectangle (inset box)
        cv2.rectangle(
            frame,
            (INSET_X, INSET_Y),
            (INSET_X + INSET_W, INSET_Y + INSET_H),
            (255, 255, 255), 2,
        )
 
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            pointer = hand[POINTER_LANDMARK]
            rx = remap(pointer.x, X_LOW, X_HIGH)
            ry = remap(pointer.y, Y_LOW, Y_HIGH)
 
            sx = smooth(sx, rx, ALPHA)
            sy = smooth(sy, ry, ALPHA)
 
            raw_history.append(rx)
            smooth_history.append(sx)
 
            ratio = pinch_ratio(hand, w, h)
            pinched = update_pinch(pinched, ratio, PINCH_ON, PINCH_OFF)
 
            # ---- drive the real mouse ----
            if control_enabled:
                move(sx, sy)
                if pinched != button_down:  # only send on a change
                    button_down = pinched
                    click(button_down)
 
            # Raw position in the inset (red) vs smoothed position (yellow/magenta).
            raw_dot = (int(INSET_X + rx * INSET_W), int(INSET_Y + ry * INSET_H))
            dot = (int(INSET_X + sx * INSET_W), int(INSET_Y + sy * INSET_H))
            cv2.circle(frame, raw_dot, 4, (0, 0, 255), -1)
            dot_color = (255, 0, 255) if pinched else (0, 255, 255)
            cv2.circle(frame, dot, 5, dot_color, -1)
 
            # Dot on the pointer landmark: green normally, red while pinched.
            pointer_px = (int(pointer.x * w), int(pointer.y * h))
            pointer_color = (0, 0, 255) if pinched else (0, 255, 0)
            cv2.circle(frame, pointer_px, 10, pointer_color, -1)
 
            cv2.putText(frame, f"Pinch Ratio: {ratio:.2f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if pinched:
                cv2.putText(frame, "PINCHED", (10, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
 
            # Jitter measurement (needs at least 2 samples). Scaled x1000 for readability.
            if len(raw_history) > 1:
                raw_jitter = statistics.pstdev(raw_history) * 1000
                smooth_jitter = statistics.pstdev(smooth_history) * 1000
                cv2.putText(frame, f"raw jitter:    {raw_jitter:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"smooth jitter: {smooth_jitter:.2f}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            # Hand lost: start fresh when it comes back, and drop stale samples.
            sx = None
            sy = None
            raw_history.clear()
            smooth_history.clear()
            pinched = False
            # Never leave the mouse button held when the hand disappears.
            if button_down:
                button_down = False
                click(False)
 
        # FPS calculation and Display
        elapsed_time = time.time() - previous_time
        if elapsed_time > 0:
            fps_text = str(round(1/elapsed_time, 2)) + " fps"
        else:
            fps_text = "0 fps"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
        # Control status
        status = "CONTROL ON  (c = off)" if control_enabled else "CONTROL OFF (c = on)"
        status_color = (0, 255, 0) if control_enabled else (0, 165, 255)
        cv2.putText(frame, status, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
 
        # Show the frame after modification.
        cv2.imshow("Camera", frame)
 
        # Keys: q = quit, c = toggle mouse control.
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord("q"):
            break
        elif key == ord("c"):
            control_enabled = not control_enabled
            if not control_enabled and button_down:
                button_down = False
                click(False)
finally:
    # Always release the button and remove the virtual device, even on a crash.
    if button_down:
        click(False)
    ui.close()
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
