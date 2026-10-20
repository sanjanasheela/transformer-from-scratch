import torch
import torch.nn as nn
import math
import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset


class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        # x = x.long()   # for an error fix 
        return self.embedding(x) * math.sqrt(self.d_model) 


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # We create a matrix of [seq_len, d_model] representing the positional encoding for each position and dimension
        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # Shape: [seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even position and cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: [1, seq_len, d_model]
        
        self.register_buffer('pe', pe) # to store pe with the model state
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation.
    RoPE applies rotation matrices to query and key vectors instead of adding positional encodings.
    """
    
    def __init__(self, d_model, seq_len, dropout=0.1, base=10000) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.base = base
        
        # RoPE works on pairs of dimensions, so d_model should be even
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        # Calculate the rotation frequencies
        # theta_i = base^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute positional encodings for efficiency
        self._build_cache(seq_len)
    
    def _build_cache(self, seq_len):
        """Precompute sin and cos values for all positions"""
        # Create position indices
        position = torch.arange(seq_len).float()  # [seq_len]
        
        # Calculate theta = position * inv_freq for all position-frequency pairs
        # position: [seq_len, 1], inv_freq: [d_model//2] -> theta: [seq_len, d_model//2]
        theta = torch.outer(position, self.inv_freq)
        
        # Calculate sin and cos
        cos_cache = torch.cos(theta)  # [seq_len, d_model//2]
        sin_cache = torch.sin(theta)  # [seq_len, d_model//2]
        
        # Register as buffers so they move with the model
        self.register_buffer('cos_cache', cos_cache)
        self.register_buffer('sin_cache', sin_cache)
    
    def apply_rotary_pos_emb(self, x, start_pos=0):
        """
        Apply rotary positional embedding to input tensor x
        Args:
            x: input tensor of shape [batch, num_heads, seq_len, d_k]
            start_pos: starting position (useful for inference)
        Returns:
            Tensor with RoPE applied
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        
        # Ensure d_k matches our model dimension
        assert d_k == self.d_model, f"Input d_k ({d_k}) must match RoPE d_model ({self.d_model})"
        
        # Get the relevant sin/cos values for this sequence
        cos = self.cos_cache[start_pos:start_pos + seq_len]  # [seq_len, d_k//2]
        sin = self.sin_cache[start_pos:start_pos + seq_len]  # [seq_len, d_k//2]
        
        # Reshape x to separate even and odd dimensions
        # x: [batch, num_heads, seq_len, d_k] -> [batch, num_heads, seq_len, d_k//2, 2]
        x_reshaped = x.view(batch_size, num_heads, seq_len, d_k // 2, 2)
        
        # Extract even and odd parts
        x_even = x_reshaped[..., 0]  # [batch, num_heads, seq_len, d_k//2]
        x_odd = x_reshaped[..., 1]   # [batch, num_heads, seq_len, d_k//2]
        
        # Apply rotation: [cos -sin; sin cos] * [x_even; x_odd]
        # Broadcasting cos/sin from [seq_len, d_k//2] to [batch, num_heads, seq_len, d_k//2]
        rotated_even = cos.unsqueeze(0).unsqueeze(0) * x_even - sin.unsqueeze(0).unsqueeze(0) * x_odd
        rotated_odd = sin.unsqueeze(0).unsqueeze(0) * x_even + cos.unsqueeze(0).unsqueeze(0) * x_odd
        
        # Recombine even and odd parts
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)  # [batch, num_heads, seq_len, d_k//2, 2]
        rotated = rotated.view(batch_size, num_heads, seq_len, d_k)  # [batch, num_heads, seq_len, d_k]
        
        return rotated
    
    def forward(self, x):
        """
        For compatibility with PositionalEncoding interface.
        Note: In RoPE, this doesn't actually apply the rotation - that happens in attention.
        This just applies dropout and returns the input.
        """
        return self.dropout(x)


class RelativePositionalBias(nn.Module):
    """
    Relative Position Bias implementation.
    Adds learnable bias terms to attention scores based on relative positions.
    """
    
    def __init__(self, num_heads, seq_len, dropout=0.1, clamp_max_distance=32):
        super().__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.clamp_max_distance = clamp_max_distance
        
        # Number of relative position buckets
        # We use 2 * clamp_max_distance + 1 buckets to cover all relative distances
        # from -clamp_max_distance to +clamp_max_distance
        num_buckets = 2 * clamp_max_distance + 1
        
        # Learnable bias table
        # Shape: [num_heads, num_buckets]
        self.relative_attention_bias = nn.Parameter(
            torch.zeros(num_heads, num_buckets)
        )
        
        # Initialize with small random values
        nn.init.normal_(self.relative_attention_bias, mean=0.0, std=0.02)
        
        # Precompute the bias matrix for efficiency
        self._build_bias_cache(seq_len)
    
    def _get_relative_position_bucket(self, relative_position):
        """
        Convert relative positions to bucket indices.
        Clamps distances beyond clamp_max_distance.
        """
        # Clamp the relative positions to the maximum distance
        relative_position = torch.clamp(
            relative_position, 
            min=-self.clamp_max_distance, 
            max=self.clamp_max_distance
        )
        
        # Convert to bucket indices (shift by clamp_max_distance to make non-negative)
        bucket_indices = relative_position + self.clamp_max_distance
        
        return bucket_indices
    
    def _build_bias_cache(self, seq_len):
        """Precompute the bias matrix for all position pairs"""
        # Create relative position matrix
        # positions: [seq_len]
        positions = torch.arange(seq_len)
        
        # relative_positions: [seq_len, seq_len]
        # relative_positions[i, j] = j - i (position j relative to position i)
        relative_positions = positions[None, :] - positions[:, None]
        
        # Get bucket indices
        bucket_indices = self._get_relative_position_bucket(relative_positions)
        
        # Register as buffer so it moves with the model
        self.register_buffer('bucket_indices', bucket_indices)
    
    def get_bias_matrix(self, seq_len_q, seq_len_k):
        """
        Get the bias matrix for the given sequence lengths.
        
        Args:
            seq_len_q: Length of query sequence
            seq_len_k: Length of key sequence
            
        Returns:
            Bias matrix of shape [num_heads, seq_len_q, seq_len_k]
        """
        # If different from cached size, recompute
        if seq_len_q != self.seq_len or seq_len_k != self.seq_len:
            # Create relative position matrix for current sizes
            positions_q = torch.arange(seq_len_q, device=self.relative_attention_bias.device)
            positions_k = torch.arange(seq_len_k, device=self.relative_attention_bias.device)
            
            # relative_positions[i, j] = j - i
            relative_positions = positions_k[None, :] - positions_q[:, None]
            
            # Get bucket indices
            bucket_indices = self._get_relative_position_bucket(relative_positions)
        else:
            # Use cached bucket indices
            bucket_indices = self.bucket_indices[:seq_len_q, :seq_len_k]
        
        # Gather bias values
        # bucket_indices: [seq_len_q, seq_len_k]
        # relative_attention_bias: [num_heads, num_buckets]
        # We need to expand bucket_indices for each head and gather
        bias = torch.zeros(self.num_heads, seq_len_q, seq_len_k, device=self.relative_attention_bias.device)
        for head_idx in range(self.num_heads):
            bias[head_idx] = self.relative_attention_bias[head_idx][bucket_indices]
        
        return bias
    
    def forward(self, x):
        """
        For compatibility with other positional encoding interfaces.
        Relative position bias is applied directly in attention, so this just applies dropout.
        """
        return self.dropout(x)


class LayerNorm(nn.Module):
    # every x = x - mean / sqrt(var + eps) 
    # to let the model to learn to scale and shift the normalized value if required we have alpha and beta as
    # alpha*x + beta
    
    def __init__(self, eps = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones((1,)))
        self.beta = nn.Parameter(torch.zeros((1,)))
            
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.alpha * x + self.beta


class feedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 B1
        self.linear2 = nn.Linear(d_ff, d_model) # W2 B2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, rope=None, relative_bias=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope  # RoPE instance, None for vanilla attention
        self.relative_bias = relative_bias  # RelativePositionalBias instance
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads  # Dimension of each head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(self, query, key, value, mask, dropout: nn.Dropout, relative_bias_matrix=None):
        d_k = query.shape[-1]
        
        # (batch, num_heads, seq_len, d_k) --> (batch, num_heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  
        
        # Add relative position bias if provided
        if relative_bias_matrix is not None:
            # relative_bias_matrix: [num_heads, seq_len_q, seq_len_k]
            # attention_scores: [batch, num_heads, seq_len_q, seq_len_k]
            attention_scores = attention_scores + relative_bias_matrix.unsqueeze(0)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        output = (attention_scores @ value)  # (batch_size, num_heads, seq_len, d_k)
        return output, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch_size, seq_len, d_model)
        key = self.w_k(k)    # (batch_size, seq_len, d_model)
        value = self.w_v(v)  # (batch_size, seq_len, d_model)   
        
        # we divide qkv into h matrices of dimension d_k each, each division will have the complete sentence but with some of the dimensions
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)        # (batch_size, num_heads, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)    # (batch_size, num_heads, seq_len, d_k)

        # Apply RoPE to query and key AFTER splitting into heads
        if self.rope is not None:
            query = self.rope.apply_rotary_pos_emb(query)
            key = self.rope.apply_rotary_pos_emb(key)

        # Get relative position bias matrix if available
        relative_bias_matrix = None
        if self.relative_bias is not None:
            seq_len_q, seq_len_k = query.shape[2], key.shape[2]
            relative_bias_matrix = self.relative_bias.get_bias_matrix(seq_len_q, seq_len_k)

        x, self.attention_scores = MultiHeadAttention.attention(self, query, key, value, mask, self.dropout, relative_bias_matrix)
        
        # (batch_size, num_heads, seq_len, d_k) --> (batch_size, seq_len, num_heads, d_k) --> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return self.w_o(x)  


class ResidualConnection(nn.Module):
    def __init__(self, dropout) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # mapping back the output embeddings to vocab size
        return torch.log_softmax(self.proj(x), dim=-1)  


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)
        
        # Filter out data points that exceed sequence length
        self.valid_data = self._filter_valid_data(ds)
        print(f"Filtered dataset: {len(self.valid_data)}/{len(ds)} samples kept "
              f"({len(self.valid_data)/len(ds)*100:.1f}% retention rate)")
        
    def _filter_valid_data(self, ds):
        """Filter out data points that exceed the sequence length."""
        valid_data = []
        skipped_count = 0
        
        for item in ds:
            src_text = item['translation'][self.src_lang]
            tgt_text = item['translation'][self.tgt_lang]
            
            # Tokenize to check lengths
            enc_input_tokens = self.tokenizer_src.encode(src_text).ids
            dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
            
            # Check if they fit within sequence length (accounting for special tokens)
            enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for SOS and EOS
            dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # -1 for SOS (EOS is in label)
            
            if enc_num_padding_tokens >= 0 and dec_num_padding_tokens >= 0:
                valid_data.append(item)
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} samples that exceeded sequence length {self.seq_len}")
            print(f"  - Samples too long for encoder: {sum(1 for item in ds if self.seq_len - len(self.tokenizer_src.encode(item['translation'][self.src_lang]).ids) - 2 < 0)}")
            print(f"  - Samples too long for decoder: {sum(1 for item in ds if self.seq_len - len(self.tokenizer_tgt.encode(item['translation'][self.tgt_lang]).ids) - 1 < 0)}")
        
        return valid_data
        
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, index):
        src_target_pair = self.valid_data[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        # This should never happen now due to filtering, but keep as safety check
        assert enc_num_padding_tokens >= 0 and dec_num_padding_tokens >= 0, \
            f"Sequence length validation failed after filtering: enc_pad={enc_num_padding_tokens}, dec_pad={dec_num_padding_tokens}"
        
        encoder_input  = torch.cat(
            [
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        decoder_input = torch.cat(
            [
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        label = torch.cat(
            [
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == self.seq_len, f"Encoder input size mismatch: {encoder_input.size(0)} vs {self.seq_len}"
        assert decoder_input.size(0) == self.seq_len, f"Decoder input size mismatch: {decoder_input.size(0)} vs {self.seq_len}"
        assert label.size(0) == self.seq_len, f"Label size mismatch: {label.size(0)} vs {self.seq_len}"
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, 1, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not Path.exists(tokenizer_path):
        print(f"Building BPE tokenizer for {lang}")
        # Initialize BPE tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Configure BPE trainer with appropriate vocabulary size
        # Vocab size of 32000 is common for BPE tokenizers
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            vocab_size=3000,
            min_frequency=2,
            continuing_subword_prefix="",
            end_of_word_suffix=""
        )
        
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        print(f"BPE tokenizer saved to {tokenizer_path}")
        
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Loaded tokenizer from {tokenizer_path}")
        
    return tokenizer


def load_custom_dataset(fi_file_path, en_file_path, save_path="datasets/custom_dataset.json"):
    """
    Load custom Finnish-English parallel dataset from text files.
    Each line in fi_file corresponds to the same line in en_file.
    Returns a list of dictionaries with format: {"translation": {"fi": "...", "en": "..."}}
    Saves the dataset to a JSON file for future use.
    """
    # Check if saved dataset already exists
    if Path(save_path).exists():
        print(f"Loading existing dataset from {save_path}")
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            print(f"Loaded {len(dataset)} translation pairs from saved file")
            return dataset
        except Exception as e:
            print(f"Error loading saved dataset: {e}. Creating new dataset...")
    
    dataset = []
    
    try:
        with open(fi_file_path, 'r', encoding='utf-8') as fi_file, \
             open(en_file_path, 'r', encoding='utf-8') as en_file:
            
            fi_lines = fi_file.readlines()
            en_lines = en_file.readlines()
            
            # Ensure both files have the same number of lines
            if len(fi_lines) != len(en_lines):
                print(f"Warning: Finnish file has {len(fi_lines)} lines, English file has {len(en_lines)} lines")
                min_lines = min(len(fi_lines), len(en_lines))
                fi_lines = fi_lines[:min_lines]
                en_lines = en_lines[:min_lines]
            
            # Create dataset entries
            for fi_line, en_line in zip(fi_lines, en_lines):
                fi_sentence = fi_line.strip()
                en_sentence = en_line.strip()
                
                # Skip empty lines
                if fi_sentence and en_sentence:
                    dataset.append({
                        "translation": {
                            "fi": fi_sentence,
                            "en": en_sentence
                        }
                    })
        
        # Save the dataset to JSON file
        try:
            # Ensure the directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"Dataset saved to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save dataset: {e}")
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []
    
    print(f"Loaded {len(dataset)} translation pairs")
    return dataset


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Perform greedy decoding to generate translation
    Works with batch_size = 1 only for simplicity
    """
    batch_size = source.size(0)
    if batch_size != 1:
        raise ValueError(f"greedy_decode only supports batch_size=1, got {batch_size}")
    
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Encode the source sequence
    encoder_output = model.encode(source, source_mask)
    
    # Initialize decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    # Add a step counter to prevent infinite loops
    step_count = 0
    max_steps = max_len * 2  # Safety limit (should be much less than this)
    
    while True:
        step_count += 1
        
        # Safety check to prevent infinite loops
        if step_count > max_steps:
            print(f"Warning: Greedy decode reached max steps ({max_steps}), stopping early")
            break
            
        if decoder_input.size(1) == max_len:
            break
            
        # Create decoder mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Decode
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # Project to vocabulary
        prob = model.project(out[:, -1])
        
        # Get next token (greedy)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)
