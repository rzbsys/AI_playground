import winsound as sd
import torch
import cv2

#모델 불러오기
model = torch.hub.load('yolov5', 'custom', path='rmask.pt', source='local')  # Epochs 500 BatchSize 32
cap = cv2.VideoCapture(0)

#비프음 정의
def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms == 1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

# 화면 크기 설정
cap.set(3,320)
cap.set(4,240)
while True:
    ret, frame = cap.read()
    if ret:
        result = model(frame)
        for i in result.xyxy:
            for x1, y1, x2, y2, conf, cls in i:
                x1, y1, x2, y2, conf, cls = int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)
                if conf < 0.5:
                    continue
                print(x1, y1, x2, y2, conf, cls)
                col = ''
                if cls == 1:
                    col = (0, 0, 255)   #red
                    beepsound()
                else:
                    col = (0, 255, 0)   #green
                reframe = cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
        cv2.imshow('video', reframe)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break
cap.release()
cv2.destroyAllWindows()
