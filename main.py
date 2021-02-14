from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import numpy as np
import time
import torch
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import math, time
from sklearn.metrics import mean_squared_error

print('loading')
csv = pd.read_csv('btc.csv')
print('loaded')
csv = csv.sort_values('Timestamp')
csv = csv.dropna()
csv = csv.reset_index()
csv = csv[4000:6000]
csv = csv.reset_index()

price = csv['Close']
scaler = MinMaxScaler(feature_range=(-1, 1))
price = scaler.fit_transform(price.values.reshape(-1,1))

def split_data(stock, lookback):
  data_raw = np.array(stock)# convert to numpy array
  data = []
  
  # create all possible sequences of length seq_len
  for index in range(len(data_raw) - lookback): 
      data.append(data_raw[index: index + lookback])
  
  data = np.array(data);
  test_set_size = int(np.round(0.2*data.shape[0]));
  train_set_size = data.shape[0] - (test_set_size);
  
  x_train = data[:train_set_size,:-1,:]
  y_train = data[:train_set_size,-1,:]
  
  x_test = data[train_set_size:,:-1]
  y_test = data[train_set_size:,-1,:]
  
  return [x_train, y_train, x_test, y_test]

lookback = 20
x_train, y_train, x_test, y_test = split_data(price, lookback)

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

input_dim = 1
hidden_dim = 42
num_layers = 1
output_dim = 1
num_epochs = 40

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x.float(), (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = .5
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

for t in range(num_epochs):
    x_train = x_train.clone().detach() 
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

minutes = np.array(price[:len(price) - 200])
model.eval()

prediction = pd.DataFrame(scaler.inverse_transform(minutes))

# make predictions
model.eval()
y_test_pred = model(x_test)
# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

sns.set_style("darkgrid")    

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 2, 1)
ax = sns.lineplot(x = csv.index[:y_train.shape[0]], y = y_train[:,0], label="Actual", color='tomato')
ax = sns.lineplot(x = csv.index[:y_train.shape[0]], y = y_train_pred[:,0], label="Train", color='darkturquoise')
ax.set_title('Train', size = 14, fontweight='bold')
ax.set_ylabel("Cost (USD)", size = 14)
ax.set_xticklabels('', size=10)

plt.subplot(1, 2, 2)
ax = sns.lineplot(x = csv.index[:y_test.shape[0]], y = y_test[:,0], label="Actual", color='tomato')
ax = sns.lineplot(x = csv.index[:y_test.shape[0]], y = y_test_pred[:,0], label="Test", color='darkturquoise')
ax.set_title('Test', size = 14, fontweight='bold')
ax.set_ylabel("Cost (USD)", size = 14)
ax.set_xticklabels('', size=10)

plt.savefig("test.png")