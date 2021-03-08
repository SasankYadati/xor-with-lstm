import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from data_helper import XORDataset, getVariableLengths
from Params import DataParams, NetworkParams, TrainingParams, Params

class Model(nn.Module):
    def __init__(self, network_params:NetworkParams):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=network_params.num_hidden_features, num_layers=network_params.num_layers, batch_first=True)
        self.fc = torch.nn.Linear(in_features=network_params.num_hidden_features, out_features=1)
    
    def forward(self, x, lengths):
        # take sequences of possibly varying lengths and form a packed tensor
        x = rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        # unpack the packed tensor back to sequences of possibly varying lengths
        x, _ = rnn.pad_packed_sequence(x, batch_first=True)
        logits = self.fc(x)
        predictions = torch.sigmoid(logits)
        return logits, predictions
    
    def evaluate(self, test_loader, is_seq_len_varying):
        is_correct = np.array([])

        for inputs, targets in test_loader:
            lengths = getVariableLengths(inputs, targets, is_seq_len_varying)

            with torch.no_grad():
                logits, predictions = self.forward(inputs, lengths)
                is_correct = np.append(is_correct, ((predictions > 0.5) == (targets > 0.5)))

        accuracy = is_correct.mean()
        return accuracy

def train_model(model:Model, params:Params, max_steps=50000):
    print(f"\nSeq len:{params.data.max_seq_len}, Varying seq len:{params.data.is_seq_len_varying}")
    train_loader = DataLoader(XORDataset(params.data.num_samples, params.data.max_seq_len), batch_size=params.data.batch_size)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.training.lr)
    step = 0
    while True:
        test_accuracy = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            lengths = getVariableLengths(inputs, targets, params.data.is_seq_len_varying)
            logits, predictions = model(inputs, lengths)
            
            loss_val = loss(logits, targets)
            loss_val.backward()
            optimizer.step()
            
            step += 1

            accuracy = ((predictions > 0.5) == (targets > 0.5)).type(torch.FloatTensor).mean()

            if step % 100 == 0:
                seq_len = params.data.max_seq_len * 2
                num_seqs = 5000
                batch_size = params.data.batch_size
                test_loader = DataLoader(XORDataset(num_seqs, seq_len), batch_size=batch_size)
                test_accuracy = model.evaluate(test_loader, params.data.is_seq_len_varying)
                
                if test_accuracy >= 0.98:
                    print(f'step {step}, loss {loss_val.item():.{4}f}, accuracy {accuracy:.{4}f}, test accuracy {test_accuracy:.{4}f}')
                    return step
            
            if step == max_steps:
                return step

if __name__ == '__main__':
    data1, data2 = DataParams(50000, 50, True), DataParams(50000, 50, False)
    network, training =  NetworkParams(), TrainingParams()
    params1 = Params(data1, network, training)
    params2 = Params(data2, network, training)
    model1 = Model(params1.network)
    model2 = Model(params2.network)
    _ = train_model(model1, params1, loss)
    _ = train_model(model2, params2, loss)
