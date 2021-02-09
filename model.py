import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.utils.checkpoint as checkpoint

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