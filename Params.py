class DataParams():
    def __init__(self, num_samples=100000, max_seq_len=50, is_seq_len_varying=False):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.is_seq_len_varying = is_seq_len_varying

    def __str__(self):
        return self.getString()
    
    def __repr__(self):
        return self.getString()

    def getString(self):
        return f"No. samples: {self.num_samples}, Max sequence length: {self.max_seq_len}, Variable length sequence: {self.is_seq_len_varying}"

class NetworkParams():
    def __init__(self, num_hidden_features=2, num_layers=1):
        self.num_hidden_features = num_hidden_features
        self.num_layers = num_layers

    def __str__(self):
        return self.getString()
    
    def __repr__(self):
        return self.getString()
    
    def getString(self):
        return f"No. hidden features: {self.num_hidden_features}, No. layers: {self.num_layers}"

class TrainingParams():
    def __init__(self, batch_size=32, lr=0.3):
        self.batch_size = batch_size
        self.lr = lr

    def __str__(self):
        return self.getString()
    
    def __repr__(self):
        return self.getString()
    
    def getString(self):
        return f"Batch size: {self.batch_size}, Learning rate: {self.lr}"


class Params():
    def __init__(self, data_params:DataParams, network_params:NetworkParams, training_params:TrainingParams):
        self.data = data_params
        self.network = network_params
        self.training = training_params
    
    def __str__(self):
        return self.getString()
    
    def __repr__(self):
        return self.getString()
    
    def getString(self):
        return f"Data params\n{self.data}\n\nNetwork params\n{self.network}\n\nTraining params\n{self.training}"