from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import math, time
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.utils.checkpoint as checkpoint

dfs = pd.read_csv('2012-01-01_to_2020-12-31.csv', sep=",", header=0,iterator=True, chunksize=100000, dtype=str)

for i,df in enumerate(dfs,start=0):
    print('Chunk: ')
    print(i)
    df = df.dropna()
    df = df.reset_index()
#     print(df)
    # do something
    # df.to_csv('output_file.csv', mode='a', index=False)

    # csv =  pd.read_csv('practice.csv')
    # csv = csv.sort_values('Timestamp')
    # csv = csv.dropna()
    # csv = csv.reset_index()

    # price = csv['Close']
    price = df['Close']

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
        
        return [x_train, y_train]

    lookback = 20
    x_train, y_train = split_data(price, lookback)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 40

    class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim, batch_size = 1000):
            super(LSTM, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.batch_size = batch_size
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def custom(self, module):
            def custom_forward(*inputs):
                inputs = module(inputs[0])
                return inputs
            return custom_forward

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, (hn, cn) = self.lstm(x.float(), (h0.detach(), c0.detach()))
            # out = checkpoint.checkpoint(self.custom(self.module), out)
            out = self.fc(out[:, -1, :]) 
            return out

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = np.zeros(num_epochs)
    start_time = time.time()

    for t in range(num_epochs):
        epoch_time = time.time()

        x_train = x_train.clone().detach() 
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train)
        # print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.save({
        'epoch': t,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, './weights.pth')

        # print("Epoch time: {}".format(epoch_time))
        
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))