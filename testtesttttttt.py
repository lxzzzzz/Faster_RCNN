import torch
from torch import nn


# hyper parameters
in_dim=1
n_hidden_1=1
n_hidden_2=1
out_dim=1

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), 
            nn.ReLU(True)
            )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            )
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
            
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
      
        
model = Net(in_dim, n_hidden_1, n_hidden_2, out_dim)
print([i for i in model.parameters()])
# print([(name, param.shape) for name, param in model.named_parameters()])

# print(model.layer1)


for layer in model.children():
    print(layer)