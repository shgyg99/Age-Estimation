from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from functions import dataset_to_csv, UTKDataset
from transforms import train_transform, test_transform

dataset = dataset_to_csv('E:\\My Drive\\deepcatalist\\utkcropped\\')

train, df_test = train_test_split(dataset, test_size=0.3, stratify=dataset['age'])
df_train, df_valid = train_test_split(train, test_size=0.2, stratify=train['age'])

df_train.to_csv('E:\\My Drive\\deepcatalist\\train_set.csv', index=False)
df_valid.to_csv('E:\\My Drive\\deepcatalist\\valid_set.csv', index=False)
df_test.to_csv('E:\\My Drive\\deepcatalist\\test_set.csv', index=False)

train = UTKDataset('E:\\My Drive\\deepcatalist\\utkcropped\\', 'E:\\My Drive\\deepcatalist\\train_set.csv', train_transform)
valid = UTKDataset('E:\\My Drive\\deepcatalist\\utkcropped\\', 'E:\\My Drive\\deepcatalist\\valid_set.csv', test_transform)
test = UTKDataset('E:\\My Drive\\deepcatalist\\utkcropped\\', 'E:\\My Drive\\deepcatalist\\test_set.csv', test_transform)

train_loader = DataLoader(train, 32, True)
valid_loader = DataLoader(valid, 64, False)
test_loader = DataLoader(test, 64, False)
