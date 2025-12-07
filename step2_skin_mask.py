import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Convert to YCrCb color space (good for skin detection)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # Skin color range (you can tweak later if needed)
        lower = np.array([0, 133, 77])
        upper = np.array([255, 173, 127])

        mask = cv2.inRange(ycrcb, lower, upper)

        # Smooth mask to reduce noise
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        cv2.imshow("Step 2 - Original", frame)
        cv2.imshow("Step 2 - Skin Mask", mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
