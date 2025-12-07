import cv2
import numpy as np

# Virtual boundary (a box on the screen)
box_top_left = (300, 100)
box_bottom_right = (500, 300)

SAFE_TH = 120   # pixels
DANGER_TH = 50  # pixels

def detect_hand_point(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 2000:
        return None

    hull = cv2.convexHull(c)
    topmost = tuple(hull[hull[:, :, 1].argmin()][0])
    return topmost, hull

def distance_to_box(point):
    (x, y) = point
    (x1, y1) = box_top_left
    (x2, y2) = box_bottom_right

    # If inside the box → distance 0
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 0

    # Distance from point to rectangle
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return int(np.sqrt(dx*dx + dy*dy))

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
        h, w = frame.shape[:2]

        # Skin detection
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77])
        upper = np.array([255, 173, 127])
        mask = cv2.inRange(ycrcb, lower, upper)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        detection = detect_hand_point(mask)
        state = "SAFE"

        if detection is not None:
            fingertip, hull = detection

            cv2.drawContours(display, [hull], -1, (0, 255, 0), 2)
            cv2.circle(display, fingertip, 8, (0, 0, 255), -1)

            dist = distance_to_box(fingertip)

            if dist <= DANGER_TH:
                state = "DANGER"
            elif dist <= SAFE_TH:
                state = "WARNING"
            else:
                state = "SAFE"

        # Draw virtual box
        cv2.rectangle(display, box_top_left, box_bottom_right, (255, 0, 0), 2)

        # State display
        if state == "SAFE":
            cv2.putText(display, "SAFE", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        elif state == "WARNING":
            cv2.putText(display, "WARNING", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        elif state == "DANGER":
            cv2.putText(display, "DANGER DANGER", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # Red border around whole screen
            cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 8)

        cv2.imshow("Step 4 - Full Prototype", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
