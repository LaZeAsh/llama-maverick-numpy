import numpy as np
from typing import Optional, Tuple

class GroupQueryAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
    ):
        """
        Group Query Attention for Llama 4 Maverick
        
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (fewer than query heads)
            head_dim: Dimension of each attention head (if None, computed from hidden_size / num_heads)
            dropout_rate: Attention dropout rate
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.dropout_rate = dropout_rate
        
        self.num_queries_per_kv = num_heads // num_kv_heads
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.q_proj = np.random.normal(0, 0.02, (hidden_size, num_heads * self.head_dim))
        self.k_proj = np.random.normal(0, 0.02, (hidden_size, num_kv_heads * self.head_dim))
        self.v_proj = np.random.normal(0, 0.02, (hidden_size, num_kv_heads * self.head_dim))
        self.o_proj = np.random.normal(0, 0.02, (num_heads * self.head_dim, hidden_size))
        
    def forward(
        self, 
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        past_key_value: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        """
        Forward pass for Group Query Attention
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask tensor of shape [batch_size, 1, seq_len, seq_len]
            position_ids: Position ids of shape [batch_size, seq_len]
            past_key_value: Cached key and value states for autoregressive generation
            output_attentions: Whether to output attention weights
            use_cache: Whether to use cached key/values for generation
            
        Returns:
            output: Output tensor of shape [batch_size, seq_len, hidden_size]
            present_key_value: Updated key/value states (if use_cache is True)
            attention_weights: Attention weights (if output_attentions is True)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to queries, keys, values

        # [batch_size, seq_len, num_heads * head_dim]
        q = hidden_states @ self.q_proj  # Each head has it's own Query
        # [batch_size, seq_len, num_kv_heads * head_dim]
        k = hidden_states @ self.k_proj  # Only need enough Keys and Values for the number of "Groups"
        v = hidden_states @ self.v_proj
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Handle cached key/values for generation
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = np.concatenate([past_k, k], axis=1)
            v = np.concatenate([past_v, v], axis=1)
        
        # Save present key/value for cache
        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None
            
        # Transpose for batched matrix multiplication
        q = q.transpose(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(0, 2, 1, 3)  # [batch_size, num_kv_heads, seq_len, head_dim]
        v = v.transpose(0, 2, 1, 3)  # [batch_size, num_kv_heads, seq_len, head_dim]
        
        # Repeat k and v for all query heads
        if self.num_kv_heads < self.num_heads:
            # For each kv head, repeat to match the query head groups
            k = np.repeat(k, self.num_queries_per_kv, axis=1)  # Now has shape [batch_size, num_heads, seq_len, head_dim]
            v = np.repeat(v, self.num_queries_per_kv, axis=1)  # Now has shape [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        attention_scores = np.matmul(q, k.transpose(0, 1, 3, 2))  # [batch_size, num_heads, seq_len, kvseq_len]
        attention_scores = attention_scores / np.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Apply softmax to get attention weights
        attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / (np.sum(attention_weights, axis=-1, keepdims=True) + 1e-6)
        
        # Apply dropout
        if self.dropout_rate > 0 and self.training:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout_rate)
        
        # Apply attention to values
        context = np.matmul(attention_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back
        context = context.transpose(0, 2, 1, 3)  # [batch_size, seq_len, num_heads, head_dim]
        context = context.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, num_heads * head_dim]
        
        # Output projection
        output = context @ self.o_proj  # [batch_size, seq_len, hidden_size]
        
        outputs = (output, present_key_value)
        if output_attentions:
            outputs += (attention_weights,)
            
        return outputs


class RoPEPositionalEncoding:
    """Rotary Positional Encoding for Llama 4 Maverick"""
    
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: int = 10000):
        """
        Initialize RoPE positional encoding
        
        Args:
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            base: Base value for frequency calculation
        """
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        # Create frequency matrix
        freqs = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
        
        # Create position matrix
        positions = np.arange(max_seq_len)
        
        # Create position frequency matrix
        freqs = positions[:, None] * freqs[None, :]  # [seq_len, head_dim/2]
        
        # Create complex rotation matrix
        self.cos_cached = np.cos(freqs)  # [seq_len, head_dim/2]
        self.sin_cached = np.sin(freqs)  # [seq_len, head_dim/2]
    
    def apply_rotary_embeddings(self, x: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        """
        Apply rotary positional embeddings to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_heads, head_dim]
            position_ids: Position ids of shape [batch_size, seq_len]
            
        Returns:
            x_rotated: Tensor with positional information
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        
        # Get position-specific sinusoidal values
        cos = self.cos_cached[position_ids]  # [batch_size, seq_len, head_dim/2]
        sin = self.sin_cached[position_ids]  # [batch_size, seq_len, head_dim/2]
        
        # Reshape and duplicate for the entire head dimension
        cos = np.repeat(cos[:, :, None, :], num_heads, axis=2)  # [batch_size, seq_len, num_heads, head_dim/2]
        sin = np.repeat(sin[:, :, None, :], num_heads, axis=2)  # [batch_size, seq_len, num_heads, head_dim/2]
        
        # Pad to match head dimension if needed
        if cos.shape[-1] < head_dim // 2:
            padding = head_dim // 2 - cos.shape[-1]
            cos = np.pad(cos, ((0, 0), (0, 0), (0, 0), (0, padding)))
            sin = np.pad(sin, ((0, 0), (0, 0), (0, 0), (0, padding)))
        
        # Concatenate to match head dimension
        cos = np.concatenate([cos, cos], axis=-1)  # [batch_size, seq_len, num_heads, head_dim]
        sin = np.concatenate([sin, sin], axis=-1)  # [batch_size, seq_len, num_heads, head_dim]
        
        # Split into even and odd dimensions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # Apply rotary transformation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_odd * cos + x_even * sin
        
        # Interleave the results
        x_rotated = np.zeros_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        
        return x_rotated 