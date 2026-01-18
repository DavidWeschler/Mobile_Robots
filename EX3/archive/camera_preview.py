import cv2

CAMERA_INDEX = 2  # might 0, 1 or 2 (phone link is usually 2)
BACKEND = cv2.CAP_MSMF  # CAP_DSHOW or CAP_MSMF

cap = cv2.VideoCapture(CAMERA_INDEX, BACKEND)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open camera with MSMF backend")

print("📱 Phone camera (DroidCam) active via MSMF")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame grab failed")
        break

    cv2.imshow("Phone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # if user presses 's', save the current frame
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("camera_snapshot.png", frame)
        print("💾 Snapshot saved as camera_snapshot.png")

cap.release()
cv2.destroyAllWindows()
