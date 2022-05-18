from runVersionsEncoderForecasterWithTraningLoop import Forecaster
from baseline import baseline
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Forecaster(12, baseline, num_blocks=2, lstm_kwargs={'k': 3}).to(device)
model.load_state_dict(torch.load("../trainedModels/wave/baseline/horizon-20-21/baseline.pt"))
model.eval()

input_data = torch.zeros([64, 40, 32, 32])
output = model(input_data, horizon = 20)
print(output.shape)