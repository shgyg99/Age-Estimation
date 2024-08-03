# """Webcam Function"""
import cv2
import torch
import face_detection
from torchvision.transforms import functional as TF

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def draw_faces(im, bboxes):
    model_path = 'E:\\My Drive\\deepcatalist\\model.pt'
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)
        img = TF.to_tensor(im[x0:x1, y0:y1]).unsqueeze(0)
        age = int(model(img).item())
        label = f'age:{age}'
        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(im, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
def Webcam():
    detector = face_detection.build_detector(
          "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():


        if cv2.waitKey(2) & 0xFF == ord('s'):
            while (cv2.waitKey(1) & 0xFF == ord('q')) == False:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame,
                    'PRESS s to start',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (204, 0, 102),
                    1,
                    cv2.LINE_4)
                cv2.putText(frame,
                    'PRESS q to quite',
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (204, 0, 102),
                    1,
                    cv2.LINE_4)
                dets = detector.detect(
                        frame[:, :, ::-1]
                    )[:, :4]
                draw_faces(frame, dets)
                cv2.imshow('video',frame)


        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame,
                    'PRESS s to start',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (204, 0, 102),
                    1,
                    cv2.LINE_4)
            cv2.putText(frame,
                    'PRESS q to quite',
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (204, 0, 102),
                    1,
                    cv2.LINE_4)
        cv2.imshow('video',frame)

    cap.release()
    cv2.destroyAllWindows()
