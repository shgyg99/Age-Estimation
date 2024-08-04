import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import tqdm
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dataset_to_csv(dataset_folder):

    image_name = os.listdir(dataset_folder)
    age = [int(i.split('_')[0]) for i in image_name]
    gender = [['Male', 'Female'][int(i.split('_')[1])] for i in image_name]
    ethnicity = [['White', 'Black', 'Asian', 'Indian', 'Others'][int(i.split('_')[2])] for i in image_name]
    dataset = pd.DataFrame({'image_name':image_name, 'age':age, 'ethnicity':ethnicity, 'gender':gender})
    dataset.to_csv('E:\\My Drive\\deepcatalist\\utkface_dataset.csv')
    return pd.read_csv('E:\\My Drive\\deepcatalist\\utkface_dataset.csv')

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
