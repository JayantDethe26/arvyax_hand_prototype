import cv2

def main():
    cap = cv2.VideoCapture(0)  # 0 = default camera

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip horizontally so it's like a mirror
        frame = cv2.flip(frame, 1)

        cv2.imshow("Step 1 - Webcam", frame)

        # Press ESC (27) to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
