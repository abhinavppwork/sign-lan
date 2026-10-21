# SignBridge AI

SignBridge AI is a browser-based ASL alphabet recognition project. It uses a webcam, hand landmark tracking, a custom YOLO classifier, and text-to-speech to recognize a held sign and speak the detected letter.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/UI-Flask-000000?logo=flask)
![MediaPipe](https://img.shields.io/badge/hand_tracking-MediaPipe-0077C0)

## Features

- Live browser dashboard for the webcam feed
- Automatic hand tracking anywhere in the camera frame
- Face-free classification: only the hand crop is sent to the sign classifier
- Temporal smoothing to reduce flickering predictions
- Configurable hold timer before a letter is accepted
- Text-to-speech for each confirmed letter
- Recently spoken letter history, FPS, and recognition progress in the UI

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Flask server, camera processing, hand tracking, recognition, and speech |
| `templates/index.html` | Browser dashboard |
| `best.pt` | Custom trained ASL alphabet YOLO model |
| `hand_landmarker.task` | MediaPipe hand landmark model |

## Requirements

- Python 3.11 or newer
- A webcam
- Windows is recommended for the current `pyttsx3` speech setup

Install the required packages in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install ultralytics opencv-python pyttsx3 flask
python -m pip install mediapipe --no-deps
python -m pip install absl-py flatbuffers sounddevice cffi
```

`mediapipe --no-deps` intentionally preserves the installed OpenCV package. On some Windows setups, MediaPipe's bundled OpenCV dependency can conflict with or lock the active OpenCV binary.

## Run locally

```powershell
.\.venv\Scripts\python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser, then select **Start camera**.

## How recognition works

1. MediaPipe identifies the 21 landmarks of one hand.
2. The app creates a padded crop around that hand, regardless of where it is in the frame.
3. The YOLO model classifies the cropped sign as an A–Z letter.
4. Recent predictions are combined over a short window to reject noisy single frames.
5. The same letter must remain stable for `1.5` seconds before it is spoken.

This makes recognition much less dependent on hand placement or camera distance than a fixed detection box.

## Tips for better recognition

- Use one hand at a time and keep it fully visible.
- Use an uncluttered, well-lit background.
- Hold the final hand shape still while the progress bar completes.
- Keep the palm and fingers facing the camera when the sign requires it.
- If a sign is consistently confused, it is a limitation of the training data in `best.pt`; adding representative examples and retraining is the long-term fix.

## Configuration

You can tune these constants at the top of `app.py`:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `CONFIDENCE` | `0.55` | Minimum YOLO confidence required for a sign |
| `STABLE_SECONDS` | `1.5` | Time a sign must be held before speech |
| `HOLD_SECONDS` | `0.85` | Time to tolerate a brief lost hand/prediction |
| `SWITCH_SECONDS` | `0.60` | Time a new sign must persist before replacing the old one |

## Troubleshooting

**Camera does not start**

Close applications that may already be using the webcam, grant camera permission to Python/browser if requested, then restart the app.

**No speech**

Check the Windows volume and default playback device. The app queues speech in a background worker so each recognized letter is spoken in order.

**Hand is detected but the letter is wrong**

Hand tracking and alphabet classification are separate. The tracker can reliably find the hand, but final alphabet accuracy depends on the custom `best.pt` model and the similarity of your sign to its training images.

## License

Add a license before distributing or reusing this project publicly.
