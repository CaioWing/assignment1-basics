from math import sqrt
import torch
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        """
        Linear transformation module. This function should accept the following paremeters:

            `in_features`: int final dimension of the input
            `out_features`: int final dimension of the output
            `device`: torch.device | None = None Device to store the parameters on
            `dtype`: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weights = torch.nn.Parameter(
            data=torch.empty((1, self.out_features, self.in_features))
            )
        self._init_weights()

    def _init_weights(self):
        std_sqd = 2 / (self.in_features + self.out_features)
        torch.nn.init.trunc_normal_(
            tensor=self.weights,
            mean=0, 
            std=std_sqd, 
            a=-3*sqrt(std_sqd), 
            b=3*sqrt(std_sqd)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")

if __name__ == "__main__":
    input = torch.randn(1, 32, 7)
    linear = Linear(32, 16)