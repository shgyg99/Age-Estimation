
import torch
from functions import evaluate
from dataloader import test_loader
import torchmetrics as tm
from functions import AgeEstimationModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'E:\\My Drive\\deepcatalist\\model.pt'
metric=tm.MeanAbsoluteError().to(device)

model = torch.load(model_path, map_location=torch.device(device))
model.eval()

metric_test = evaluate(model, test_loader, metric)
metric_test
