import torch
import face_detection
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from transforms import test_transform
from functions import AgeEstimationModel

AgeEstimationModel()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def picture(model_path, image_path):    
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()

    detector = face_detection.build_detector(
            "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    image =cv2.imread(image_path)
    dets = detector.detect(
                            image[:, :, ::-1]
                        )[:, :4]
    for bbox in dets:
        top, right, bottom, left = [int(_) for _ in bbox]
        cv2.rectangle(image, (top, right), (bottom, left), (0, 0, 255), 2)
        image_cropped = image[right:left, top:bottom]
        image_transformed = test_transform(TF.to_pil_image(image_cropped)).unsqueeze(0)
        age = model(image_transformed)
        age = int(age.item())
        label = f'age:{age}'
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(image, label, (bottom-w, right - h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()