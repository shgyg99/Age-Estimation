from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
from functions import dataset_to_csv, UTKDataset
from transforms import train_transform, test_transform
from inputs import main_path

dataset = dataset_to_csv(os.path.join(main_path, 'utkcropped\\'))

train, df_test = train_test_split(dataset, test_size=0.3, stratify=dataset['age'])
df_train, df_valid = train_test_split(train, test_size=0.2, stratify=train['age'])

df_train.to_csv(os.path.join(main_path, '.csvFiles\\', 'train_set.csv'), index=False)
df_valid.to_csv(os.path.join(main_path, '.csvFiles\\', 'valid_set.csv'), index=False)
df_test.to_csv(os.path.join(main_path, '.csvFiles\\', 'test_set.csv'), index=False)

train = UTKDataset(os.path.join(main_path, 'utkcropped\\'),
                   os.path.join(main_path, '.csvFiles\\', 'train_set.csv'),
                   train_transform)

valid = UTKDataset(os.path.join(main_path, 'utkcropped\\'),
                   os.path.join(main_path, '.csvFiles\\', 'valid_set.csv'),
                   test_transform)

test = UTKDataset(os.path.join(main_path, 'utkcropped\\'),
                  os.path.join(main_path, '.csvFiles\\', 'test_set.csv'),
                  test_transform)

train_loader = DataLoader(train, 32, True)
valid_loader = DataLoader(valid, 64, False)
test_loader = DataLoader(test, 64, False)
