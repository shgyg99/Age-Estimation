import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.transforms import functional as TF
import tqdm
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from inputs import main_path
import face_detection

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dataset_to_csv(dataset_folder):

    image_name = os.listdir(dataset_folder)
    age = [int(i.split('_')[0]) for i in image_name]
    gender = [['Male', 'Female'][int(i.split('_')[1])] for i in image_name]
    ethnicity = [['White', 'Black', 'Asian', 'Indian', 'Others'][int(i.split('_')[2])] for i in image_name]
    dataset = pd.DataFrame({'image_name':image_name, 'age':age, 'ethnicity':ethnicity, 'gender':gender})
    dataset.to_csv(os.path.join(main_path, '.csvFiles', 'utkface_dataset.csv'))
    return pd.read_csv(os.path.join(main_path, '.csvFiles', 'utkface_dataset.csv'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def num_trainable_params(model):
  nums = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
  return nums

def show_random_samples(dataset_folder, num_samples=9):

    image_files = os.listdir(dataset_folder)
    selected_image_files = random.sample(image_files, num_samples)

    plt.figure(figsize=(10, 10))
    for idx, image_file in enumerate(selected_image_files, 1):
        image_path = os.path.join(dataset_folder, image_file)
        age, gender, ethnicity = image_file.split('_')[:3]

        image = Image.open(image_path)

        gender = 'Male' if int(gender) == 0 else 'Female'
        ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Others'][int(ethnicity)]

        plt.subplot(3, 3, idx)
        plt.imshow(image)
        plt.title(f"Age: {age}\nGender: {gender}\nEthnicity: {ethnicity}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def delete_wrong_files(dataset_folder):

    image_name = os.listdir(dataset_folder)
    count = 0
    for i in image_name:
        age = int(i.split('_')[0])
        gender = i.split('_')[1]
        ethnicity = i.split('_')[2]
        if gender not in ['0', '1'] or ethnicity not in ['0', '1', '2', '3', '4'] or age>80:
            os.remove(os.path.join(dataset_folder, i))
            count += 1
        else:
            pass
    print(f'deleted {count} images from root')

class UTKDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.gender_dict = {'Male':0, "Female":1}
        self.ethnicity_dict = {'White':0, 'Black':1, 'Asian':2, 'Indian':3, 'Others':4}

        self.file = pd.read_csv(csv_file)


    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        sample = self.file.iloc[idx, :]
        img_path = os.path.join(self.root_dir, sample.image_name)
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)
        age = torch.tensor(sample.age, dtype=torch.float32)
        gender = torch.tensor(self.gender_dict[sample.gender], dtype=torch.int32)
        ethnicity = torch.tensor(self.ethnicity_dict[sample.ethnicity], dtype=torch.int32)
        return image, age #gender, ethnicity
    
class AgeEstimationModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()
        self.fc = nn.Linear(in_features=512, out_features=1, bias=True)


    def forward(self, x):
        y = self.fc(self.model(x))
        return y

def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=None):
    model.train()
    metric.reset()
    with tqdm.tqdm(train_loader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
            if epoch is not None:
                tepoch.set_description(f"Epoch {epoch}")
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs).to(device)

            loss = loss_fn(outputs, targets.unsqueeze(1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            metric.update(outputs, targets.unsqueeze(1))
            tepoch.set_postfix(loss=metric.compute().item())
    return model, metric.compute().item()

def evaluate(model, test_loader, metric):
    model.eval()
    metric.reset()
    with tqdm.tqdm(test_loader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            metric.update(outputs, targets.unsqueeze(1))
            tepoch.set_postfix(loss=metric.compute().item())
    return metric.compute().item()

def picture_upload(model_path, image_path, transform):    
    
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
        image_transformed = transform(TF.to_pil_image(image_cropped)).unsqueeze(0)
        age = model(image_transformed)
        age = int(age.item())
        label = f'age:{age}'
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(image, label, (bottom-w, right - h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def draw_faces(im, bboxes):
    model_path = 'E:\\My Drive\\deepcatalist\\model2.pt'
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