🖐️ Hand Proximity Detection System (Without MediaPipe / OpenPose)

A real-time computer-vision prototype that detects the user’s hand and determines whether it is SAFE, WARNING, or in DANGER based on its proximity to the camera — without using any pose-detection libraries like MediaPipe or OpenPose.

This project fulfills all requirements for the Arvyax Internship Assignment (07 Dec 2025 submission).

🚀 Features
✔ Real-time hand tracking

Tracks the user's hand based on motion, not skin color — making it reliable in any lighting or background.

✔ No MediaPipe / No OpenPose

Uses only classical Computer Vision techniques such as:

Background subtraction (MOG2)

Contour detection

Convex hull & fingertip detection

Thresholding & denoising

✔ Full-screen danger detection

The entire screen serves as a virtual danger boundary.

The system classifies the state as:

🟢 SAFE – Hand far from camera

🟡 WARNING – Hand approaching camera

🔴 DANGER DANGER – Hand too close to the camera

✔ On-screen visual feedback

Large text overlay showing SAFE / WARNING / DANGER

Red flashing border during DANGER state

Real-time contour & fingertip visualization

✔ 8+ FPS CPU-only performance

Optimized for smooth execution without GPU or heavy ML models.

🧠 How It Works (Simple Explanation)

Instead of skin color detection, this project uses motion analysis:

The background subtractor (MOG2) learns what the background looks like.

Anything that moves (the hand) becomes white in the motion mask.

The largest moving object is assumed to be the hand.

A convex hull is drawn around the hand to stabilize its shape.

The area of the hand contour determines how close the hand is:

Small area → far from camera → SAFE

Medium area → WARNING

Large area → very close → DANGER

This makes the system independent of skin color, lighting, and background.

📦 Installation
pip install opencv-python numpy

▶️ Run the Program
python full_screen_danger.py


Make sure your webcam is connected.

📁 Project Structure
📦 HandProximityDetection
│
├── full_screen_danger.py      # Main prototype (final version)
├── step1_motion.py            # Test motion detection
├── step2_contour_tracking.py  # Hand tracking using contours
└── README.md                  # Documentation

🖥️ Demo Behavior
Hand Position	Display	Screen Behavior
Far from camera	SAFE	Normal view
Moving closer	WARNING	Yellow warning text
Very close to camera	DANGER DANGER	Red screen border + alert text
🔬 Techniques Used
Computer Vision

Background Subtraction (MOG2)

Gaussian Blurring

Thresholding

Contour Extraction

Convex Hull Detection

Logic

Contour area → hand distance estimation

Full-screen boundary instead of small box

State machine: SAFE → WARNING → DANGER

🎯 Why This Approach?

Skin detection fails when:

Background is skin-colored

Lighting changes

Different skin tones

Shadows or reflections

By switching to motion-based tracking, this prototype becomes:

More robust

More consistent

Environment-independent

Suitable for real engineering use cases

📝 Possible Improvements

If needed, the project can be extended with:

Audio alerts (beeps on DANGER)

Depth estimation using size + speed

Multi-hand detection

UI panel showing distance values

AR overlays (safety circles, warning glow)

A tiny custom ML classifier for hand blob verification

Just ask if you’d like any of these enhancements.

📬 Author

Developed as part of the Arvyax Internship Technical Assignment
Built using Python + OpenCV, without using MediaPipe or OpenPose.