import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    mask = fgbg.apply(frame)

    # Clean up noise
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow("Frame", frame)
    cv2.imshow("Motion Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
