import torch
from torch import optim
import torch.nn as nn
import tqdm
import torchmetrics as tm
from functions import train_one_epoch, evaluate
from dataloader import train_loader, valid_loader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

# model = AgeEstimationModel().to(device)
model = torch.load('E:\\My Drive\\deepcatalist\\model.pt', map_location=torch.device(device))

# model.model.requires_grad_(True)
# model.model.layer4.requires_grad_(True)
# model.fc.requires_grad_(True)

loss_train_hist = []
loss_valid_hist = []

best_loss_valid = torch.inf
epoch_counter = 0

lr = 0.001
wd = 1e-4
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
metric=tm.MeanAbsoluteError().to(device)
loss_fn = nn.L1Loss()

num_epochs = 10

for epoch in range(num_epochs):
  # Train
  model, loss_train = train_one_epoch(model,
                                    train_loader,
                                    loss_fn,
                                    optimizer,
                                    metric,
                                    epoch)
  # Validation
  loss_valid = evaluate(model,
                       valid_loader,
                       metric)

  loss_train_hist.append(loss_train)
  loss_valid_hist.append(loss_valid)

  if loss_valid < best_loss_valid:
    torch.save(model, '/content/drive/MyDrive/deepcatalist/model.pt')
    best_loss_valid = loss_valid
    print('model saved!')

  print(f'Valid: Loss = {loss_valid:.4}')
  print()

  epoch_counter += 1