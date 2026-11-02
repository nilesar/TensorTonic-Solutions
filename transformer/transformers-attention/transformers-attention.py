import torch
import math
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    val=Q @ K.transpose(-2,-1)
    d_k = Q.shape[-1]
    val=val/math.sqrt(d_k)
    attention_weights = torch.softmax(val, dim=-1)
    output=attention_weights @ V
    return output
    