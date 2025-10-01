from torch.nn import Module, Parameter
from torch import Tensor
import torch
from einops import einsum
from jaxtyping import Float

class SwiGLU(Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Parameter(torch.empty((d_ff, d_model)))
        self.w2 = Parameter(torch.empty((d_model, d_ff)))
        self.w3 = Parameter(torch.empty((d_ff, d_model)))
        
    def _SiLU(self, x: Float[Tensor, "... d_ff"]) -> Float[Tensor, "... d_ff"]:
        return x * torch.sigmoid(x)
        
    def forward(self, in_features: Float[Tensor, "... d_model"]):
        gate = einsum(in_features, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        gate = self._SiLU(gate)
        
        value = einsum(in_features, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        hidden = gate * value
        output = einsum(hidden, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        return output