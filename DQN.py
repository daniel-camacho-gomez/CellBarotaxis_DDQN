import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, hidden_layer_sizes, n_actions, init_method="default"):
        super(DQN, self).__init__()

        # Input layer with N_I input nodes and the first hidden layer.
        self.input_layer   = nn.Linear(n_observations, hidden_layer_sizes[0])  
        
        # Define the hidden layers based on the hidden_layer_sizes list.
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1])
            for i in range(len(hidden_layer_sizes) - 1)])
        
        # Output layer with the last hidden layer and N_O output nodes.
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], n_actions)  
        
        # Dropout 
        self.dropout_input_layer   = nn.Dropout(0.2) 
        self.dropout_hidden_layers = nn.Dropout(0.2) 
        
        if init_method == "xavier":
            self.init_weights_xavier_uniform()
        
    def forward(self, x):
        # Apply ReLU activation to the input layer.
        x = F.relu(self.input_layer(x))  
        # x = self.dropout_input_layer(x)
        
        # Forward pass through all the hidden layers with ReLu activation
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            # x = self.dropout_hidden_layers(x)
            
        # Output layer with tanh activation 
        x = F.tanh(self.output_layer(x))
        return x  
    
    
    def init_weights_xavier_uniform(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)