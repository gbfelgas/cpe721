import torch.nn as nn

class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        num_layers: int,
        with_softmax: bool
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for n in range(num_layers):
            if n != (num_layers - 1):
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            else:
                self.layers.append(nn.Linear(input_dim, output_dim))
                if with_softmax:
                    self.layers.append(nn.Softmax(dim=-1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x