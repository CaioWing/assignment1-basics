from torch.nn import Module
import torch

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        """
        Construct an embedding module

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimesion of the embedding vectors
            device (torch.device, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        self.weights = torch.nn.Parameter(
            data=torch.empty((num_embeddings, embedding_dim))
        )
        self._init_weights()
    
    def _init_weights(self):
        torch.nn.init.trunc_normal_(
            tensor=self.weights,
            mean=0, std=1, a=-3, b=3
            )
            
    def forward(self, token_ids: torch.Tensor):
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.weights[token_ids]