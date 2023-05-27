import torch


class PolyReLU(torch.nn.Module):
    def __init__(self, rho, alpha):
        super().__init__()
        self.rho = rho
        self.alpha = alpha

    def forward(self, x):
        x = (x < self.rho) * (x ** self.alpha) / (self.alpha * (self.rho ** (self.alpha - 1))) \
           + (x >= self.rho) * (x - (1 - 1 / self.alpha) * self.rho)

        return x


class TwoLayerCNN(torch.nn.Module):
    def __init__(self, channels, n_weights, patch_height, alpha, out_dim):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            channels,
            n_weights,
            kernel_size=patch_height,
            stride=patch_height,
            bias=False,
        )
        self.polyrelu = PolyReLU(1 / out_dim, alpha)
        self.n_classes = out_dim

        # Xavier init for weights.
        torch.nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        return torch.sum(
            self.polyrelu(self.conv(x)).view(x.shape[0], self.n_classes, -1), dim=2
        )
