import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
df = pd.read_csv("final_main_average_data.csv")
df = df[['날짜', '종가', '감성분석']]
df.set_index('날짜', inplace=True)
df = df.replace({'감성분석': {'positive': 1, 'neutral': 0, 'negative': -1}})


# Normalize data
scaler = MinMaxScaler()
df['종가'] = scaler.fit_transform(df[['종가']])

df['다음날종가'] = df['종가'].shift(-1)
df = df.drop(df.index[-1])

dfX = df.iloc[:, :2]
dfY = df.iloc[:, 2:3]

# # Test only price
# dfX.drop(columns = ['감성분석'], inplace=True)

print(dfX)
print(dfY)

plt.plot(dfY, label = 'samsung')
plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

scaleX = np.array(dfX)
scaleY = np.array(dfY)

# convert tensor
dataX = torch.FloatTensor(scaleX)
dataY = torch.FloatTensor(scaleY)

# split data
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.3, shuffle=False)


print(trainX.shape, testX.shape, trainY.shape, testY.shape)

train_size = trainX.shape[0]

# Change the shape
# reshape(a, (b,1,c)) 형태로 모델에 돌리기 위해 2차원으로 바꿔주는 것임
trainX = trainX.view(-1, 1, trainX.shape[1])
testX = testX.view(-1, 1, testX.shape[1])
dataX = dataX.view(-1, 1, dataX.shape[1])

print("Training Shape", trainX.shape, trainY.shape)
print("Testing Shape", testX.shape, testY.shape) 
print(dataX.shape, dataY.shape)

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
    

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MyModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.Tanh(),
            nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True),
        )
        self.logit_layer = nn.Linear(self.hidden_dim, 1)
  
    def forward(self, x, index=None):
        hidden_x, hn_x = self.layers(x)
        output = self.logit_layer(hn_x).squeeze()

        return output


num_epochs = 100000
learning_rate = 0.001

input_size = trainX.shape[2] # column number
hidden_size = 128
num_layers = 1

# Build model
model = LSTM(input_size, hidden_size, num_layers).to(device)
# model = MyModel(input_size, hidden_size)

# Optimizer & loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()

train_losses = []

# Train model
for epoch in range(num_epochs):
    trainX, trainY = trainX.to(device), trainY.to(device)    
    optimizer.zero_grad()

    outputs = model.forward(trainX)
    loss = loss_func(outputs, trainY)

    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # Print the loss
    if (epoch + 1) % 100 == 0:
      print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


test_preds = []

# evaluate the model on test data
with torch.no_grad():
    dataX = dataX.to(device)
    outputs = model(dataX)

    if device == 'cuda':
        test_preds += outputs.detach().cpu().numpy().tolist()
    else:
        test_preds += outputs.detach().numpy().tolist()

    print(test_preds)

    data_predict = outputs.cpu().numpy()

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Plot 
# data_predict = scaler.inverse_transform(data_predict.reshape(-1, 1))
data_predict = scaler.inverse_transform(data_predict)
dataY_plot = scaler.inverse_transform(dataY.data.numpy())

mse = mean_squared_error(dataY_plot, data_predict)
rmse = np.sqrt(mse)

mae = mean_absolute_error(dataY_plot, data_predict)
print(data_predict, mse, rmse, mae)

# dataY_plot = dataY.data.numpy()

plt.axvline(x=train_size, c='r', linestyle='--')
plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction_Model')
plt.show()