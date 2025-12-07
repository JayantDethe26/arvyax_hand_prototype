import cv2
import numpy as np

def detect_hand_point(mask):
    # Find contours on the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Assume largest contour is the hand
    c = max(contours, key=cv2.contourArea)

    # Ignore very small objects
    if cv2.contourArea(c) < 2000:
        return None

    # Convex hull for smoothing shape
    hull = cv2.convexHull(c)

    # Fingertip = point with smallest y (highest point on screen)
    topmost = tuple(hull[hull[:, :, 1].argmin()][0])

    return topmost, hull

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
        display = frame.copy()

        # Skin detection
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77])
        upper = np.array([255, 173, 127])
        mask = cv2.inRange(ycrcb, lower, upper)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        detection = detect_hand_point(mask)

        if detection is not None:
            fingertip, hull = detection

            # Draw hull (green)
            cv2.drawContours(display, [hull], -1, (0, 255, 0), 2)

            # Draw fingertip (red dot)
            cv2.circle(display, fingertip, 8, (0, 0, 255), -1)

        cv2.imshow("Step 3 - Hand + Fingertip", display)
        cv2.imshow("Step 3 - Mask", mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
