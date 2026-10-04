import cv2
import time

cap = cv2.VideoCapture(0)



while True:
    # Previous time
    previous_time = time.time()

    # Read the Camera input.
    ok, frame = cap.read()

    if not ok:
        break

    # Flip the frame horizontally for a selfie-view display.
    frame = cv2.flip(frame, 1)

    # FPS calculation and Display
    elapsed_time = time.time() - previous_time
    if elapsed_time > 0:
        fps_text = str(round(1/(time.time()-previous_time), 2)) + " fps" # 1000 ms = 1 s
    else:
        fps_text = "0 fps"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame after modification.
    cv2.imshow("Camera", frame)

    # Wait for the 'q' key to exit the loop.
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    


cap.release()
cv2.destroyAllWindows()