# """Webcam Function"""
import torch
from functions import AgeEstimationModel, Webcam

AgeEstimationModel()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Webcam()