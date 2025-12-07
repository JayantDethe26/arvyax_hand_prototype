import cv2
import numpy as np

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=16,
    detectShadows=False
)

# Thresholds for hand size (adjust if needed)
SAFE_AREA = 3000
DANGER_AREA = 12000

def get_hand(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < 1500:
        return None

    hull = cv2.convexHull(c)
    fingertip = tuple(hull[hull[:, :, 1].argmin()][0])
    
    return hull, fingertip, area


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()
    h, w = frame.shape[:2]

    # Motion mask
    mask = fgbg.apply(frame)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]

    hand_data = get_hand(mask)
    state = "SAFE"

    if hand_data:
        hull, fingertip, area = hand_data

        cv2.drawContours(display, [hull], -1, (0, 255, 0), 2)
        cv2.circle(display, fingertip, 10, (0, 0, 255), -1)

        # Full-screen danger: based on size of hand (area)
        if area > DANGER_AREA:
            state = "DANGER"
        elif area > SAFE_AREA:
            state = "WARNING"
        else:
            state = "SAFE"

    # Display SAFE / WARNING / DANGER
    if state == "SAFE":
        cv2.putText(display, "SAFE", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    elif state == "WARNING":
        cv2.putText(display, "WARNING", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)

    elif state == "DANGER":
        cv2.putText(display, "DANGER DANGER", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 8)

    cv2.imshow("Full Screen Danger System", display)
    cv2.imshow("Motion Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
