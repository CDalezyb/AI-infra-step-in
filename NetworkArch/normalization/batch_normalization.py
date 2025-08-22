import torch
import torch.nn as nn

class BatchNorm1D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.ones(num_features))
        self.reset_params()

    def reset_params(self):
        '''初始化参数'''
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.var)

    def forward(self, x):
        pass
    
def test_BatchNorm1D():
    pass

if __name__ == "__main__":
    test_BatchNorm1D()

    

