from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import math, time
from sklearn.metrics import mean_squared_error

csv = pd.read_csv('BTC-USD.csv')
# csv = csv.sort_values('Timestamp')
csv = csv.sort_values('Date')
csv = csv.dropna()
csv = csv.reset_index()
# csv = csv.drop([0,3])
# csv = csv.reset_index()
# csv = csv[1000:2000]

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
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 50

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

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
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

for i in range(200):
    start = len(minutes) - 201 + i
    end = len(minutes) - 1 + i
    last_20 = np.array([minutes[start:end]])
    last_20 = torch.from_numpy(last_20).type(torch.Tensor)
    p = model(last_20)
    p =  np.array(p.detach().numpy())
    minutes = np.array(np.append(minutes,p,axis=0))


prediction = pd.DataFrame(scaler.inverse_transform(minutes))
# print('prediction length')
# print(prediction.shape)

# predictPlot = np.empty_like(csv[['Close']])
# predictPlot[:, :] = np.nan
# predictPlot[804:1000, :] = prediction

sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(csv[['Close']])
plt.plot(prediction)
plt.xticks(range(0,csv.shape[0],500),csv['Timestamp'].loc[::500],rotation=45)
plt.title("BTC Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (USD)',fontsize=18)
plt.savefig("dummy_name.png")