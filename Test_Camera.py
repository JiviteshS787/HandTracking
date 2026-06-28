import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    success, _ = cap.read()
    print(i, success)
    cap.release()