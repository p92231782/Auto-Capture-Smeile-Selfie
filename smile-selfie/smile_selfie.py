import cv2, time, os
from datetime import datetime

cap = cv2.VideoCapture(0)
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

smile_frames, last = 0, 0
COOLDOWN, NEED = 3.0, 5

while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.2, 5, minSize=(80,80))
    smiling = False
    if len(faces):
        x,y,w,h = max(faces, key=lambda r:r[2]*r[3])
        roi = gray[y:y+h, x:x+w]
        smiles = smile.detectMultiScale(roi, 1.7, 22, minSize=(int(w*0.25), int(h*0.15)))
        smiling = len(smiles) > 0

    smile_frames = smile_frames + 1 if smiling else 0
    now = time.time()
    if smile_frames >= NEED and (now - last) > COOLDOWN:
        
        name = f"selfie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(name, frame)
        print("Saved", name)
        last, smile_frames = now, 0

    cv2.imshow("cam", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): break

cap.release(); cv2.destroyAllWindows()
