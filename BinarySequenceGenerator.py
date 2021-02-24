import random
import numpy as np

def generateBinarySequences(num_sequences, len, seed=42):
    np.random.seed(seed)
    binary_sequences = []
    parities = []
    for _ in range(num_sequences):
        binary_seq, parity = getRandomBinarySequenceWithParity(len)
        binary_sequences.append(binary_seq)
        parities.append(parity)
    return binary_sequences, parities

def getRandomBinarySequenceWithParity(len):
    parity = False
    binary_seq = np.random.randint(2, size=len)
    parity = np.logical_xor.reduce(binary_seq)
    return binary_seq, parity

if __name__ == '__main__':
    import timeit
    t = timeit.timeit(lambda: generateBinarySequences(100000, 50), number=10,  globals=globals())
    print(t)