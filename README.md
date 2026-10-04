# archle-touch
A small project involving OpenCV to control my mouse and hand movements when my mouse dies 🫣.

Control my laptop's mouse with my hand and a webcam. Point with a finger to move the cursor, pinch thumb and index finger to click and drag.

Built as a learning project. It works well enough to use, but it's not a replacement for a real mouse.

## How it works

1. OpenCV grabs frames from the webcam.
2. MediaPipe's hand landmarker finds 21 points on my hand.
3. The fingertip position inside a smaller "active rectangle" gets mapped to the whole screen, so I don't have to reach the edges of the camera view.
4. An exponential moving average smooths out the jitter.
5. Thumb-to-index distance (divided by hand size, so it works at any distance) decides whether I'm pinching. Two thresholds stop it from flickering.
6. A virtual mouse made with `evdev` / `/dev/uinput` sends the movement and clicks to the desktop.

I used `uinput` instead of something like `pyautogui` because I'm on Wayland (Arch + KDE) and X11 tools don't work there.

## Setup

Tested on Arch Linux with KDE Plasma (Wayland).

```bash
python -m venv venv
source venv/bin/activate
pip install mediapipe opencv-python evdev
```

Download the hand model into the project folder (MediaPipe 1.0 doesn't bundle it):

```bash
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Give my user access to `/dev/uinput`, then **log out and back in**:

```bash
sudo modprobe uinput
echo uinput | sudo tee /etc/modules-load.d/uinput.conf
echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' | sudo tee /etc/udev/rules.d/99-uinput.rules
sudo usermod -aG input $USER
sudo udevadm control --reload && sudo udevadm trigger
```

## Run

```bash
python main.py
```

| Key | What it does |
| --- | --- |
| `c` | Turn mouse control on/off (starts **off**) |
| `q` | Quit |

Press `c` when you're ready. Pinch to click, hold the pinch to drag. If the cursor goes somewhere weird, press `c` again or use the real mouse.

## Tuning

These are constants near the top of `main.py` and in the loop:

- `POINTER_LANDMARK`: 8 is the index fingertip. If the cursor jumps when I pinch, 5 (the knuckle) moves less.
- `ALPHA`: smoothing. Lower is steadier but laggier. I'm using 0.35.
- `PINCH_ON` / `PINCH_OFF`: pinch thresholds. Keep `OFF` well above `ON`.
- `MARGIN`: how much of the camera frame to ignore around the edges.

## Known issues

- Pinching moves the index fingertip a bit, so clicks can drift (see `POINTER_LANDMARK` above).
- Bad lighting makes the camera drop frames and tracking gets flaky. Good light matters more than any code change.
- Holding an arm up gets tiring fast.
- Only left click so far. Multi-monitor setups map across the whole desktop.

## Ideas for later

- Right click and scroll gestures
- One Euro filter instead of the plain moving average
- A "pause" gesture instead of the keyboard toggle