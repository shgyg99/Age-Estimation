import torch
import face_detection
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from transforms import test_transform
from functions import AgeEstimationModel
import os
from inputs import main_path, models
from functions import picture_upload

AgeEstimationModel()

for pic in os.listdir(os.path.join(main_path, 'test_pics\\')):
    picture = os.path.join(main_path, 'test_pics\\', pic)
    out = picture_upload(os.path.join(main_path, 'models\\', models['resnet50']), picture, test_transform)
    cv2.imwrite(os.path.join(main_path, 'test_pics\\', f'out_{pic}'), out)