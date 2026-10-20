import torch
import torch.nn as nn
from utils import MultiHeadAttention, feedForwardBlock, ResidualConnection, LayerNorm


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: feedForwardBlock, dropout) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual1 = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, mask):
        x = self.residual1[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual1[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
