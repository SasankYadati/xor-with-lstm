import random
import numpy as np
import torch
import torch.utils.data as data

class XORDataset(data.Dataset):
  def __init__(self, num_sequences, seq_len):
    self.num_sequences = num_sequences
    self.seq_len = seq_len

    self.features, self.labels = getRandomBinarySequences(num_sequences, seq_len)
    self.features = np.expand_dims(self.features, -1)
    self.labels = np.expand_dims(self.labels, -1)

  def __getitem__(self, index):
    return self.features[index, :], self.labels[index]

  def __len__(self):
    return len(self.features)

def getRandomBinarySequences(num_sequences, seq_len):
  bit_sequences = np.random.randint(2, size=(num_sequences, seq_len))
  bitsum = np.cumsum(bit_sequences, axis=1)
  parity = bitsum % 2 != 0
  return bit_sequences.astype('float32'), parity.astype('float32')

def getVariableLengths(x, y, is_seq_len_varying):
    batch_size = x.size()[0]
    max_seq_len = x.size()[1]
    if not is_seq_len_varying:
        return torch.ones(batch_size) * max_seq_len
    lengths = torch.tensor(np.random.randint(1, max_seq_len, size=batch_size))
    lengths[-1] = max_seq_len
    for i, length in enumerate(lengths):
        x[i, lengths[i]:,] = 0
        y[i, lengths[i]:,] = 0
    return lengths

if __name__ == '__main__':
    x,y = getRandomBinarySequences(2,5)
    print(x, y)