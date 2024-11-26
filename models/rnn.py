import torch
import torch.nn as nn

class SegRNN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, out_size=4, dropout=0.3):
        super(SegRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Linear(2*hidden_size, out_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.output(x)
        return x.transpose(1,2)