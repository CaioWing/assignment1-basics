import torch
from torch.nn import Module
from einops import reduce, einsum

class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """Construct the RMSNorm module. This function should accept the following parameters

        Args:
            d_model (int): Hidden dimension of the model
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weights = torch.nn.Parameter(
            torch.ones(self.d_model, device=device, dtype=dtype)
        )
    
    def _RMS(self, x : torch.Tensor):
        return torch.sqrt(
            reduce(x.square(), "... d_model -> ... 1", reduction = "mean") + self.eps
            )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        id_dtype = x.dtype
        x = x.to(torch.float32)
        return (x / self._RMS(x) * self.weights).to(id_dtype)