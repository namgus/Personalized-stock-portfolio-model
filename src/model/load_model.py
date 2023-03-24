import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LSTM(nn.Module):
    # Long Short Term Memory
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_dim, 128)
        self.fc = nn.Linear(128, 1)
        self.activation = nn.Tanh() #nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_dim)
        
        out = self.activation(h_out)
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc(out)
        
        return out

# Load the model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Scale the input data
df = pd.DataFrame({'종가': [72400]})
df = pd.DataFrame(scaler.transform(df), columns=['종가'])

# Create the input tensor
data = [{'종가': df.iloc[0, 0], '감성분석':0}]
new_data = pd.DataFrame(data)
inputs = torch.FloatTensor(new_data.values)
inputs = inputs.view(1, 1, -1).to(device)

# Make predictions and inverse transform
predictions = model(inputs)

# Convert predictions to 1D numpy array and then back to 2D
print(predictions)
predictions = predictions.detach().cpu().numpy()
predictions = np.array(predictions.squeeze())
predictions = predictions.reshape(-1, 1)

predictions = scaler.inverse_transform(predictions)

print(predictions)