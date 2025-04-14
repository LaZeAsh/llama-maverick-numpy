import numpy as np
from typing import Optional, Tuple, List, Union, Dict

from attention import GroupQueryAttention, RoPEPositionalEncoding
from ffn import FeedForwardNetwork, MixtureOfExperts

class LayerNorm:
    """Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """
        Initialize Layer Normalization
        
        Input:
            hidden_size: Hidden dimension size
            eps: Epsilon for numerical stability
        """
        self.weight = np.ones(hidden_size)
        self.bias = np.zeros(hidden_size)
        self.eps = eps
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Layer Normalization
        
        Input:
            x: Input tensor of shape [..., hidden_size]
            
        Output:
            Normalized tensor of same shape
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + self.eps)
        scaled = normalized * self.weight + self.bias
        return scaled


class TransformerBlock:
    """Transformer Block for Llama 4 Maverick"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        head_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        layer_id: int = 0,
        max_seq_len: int = 8192,
    ):
        """
        Initialize a Transformer Block
        
        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            num_kv_heads: Number of key/value heads for group query attention
            intermediate_size: Intermediate size for FFN
            use_moe: Whether to use MoE or standard FFN
            num_experts: Number of experts for MoE
            num_experts_per_token: Number of experts to route each token to
            head_dim: Dimension of each attention head 
            dropout_rate: Dropout probability
            layer_id: Layer identifier
            max_seq_len: Maximum sequence length
        """
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        
        self.input_layernorm = LayerNorm(hidden_size)
        self.post_attention_layernorm = LayerNorm(hidden_size)
        
        self.self_attn = GroupQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
        )
        
        head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.rotary_emb = RoPEPositionalEncoding(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
        )
        
        if use_moe:
            self.mlp = MixtureOfExperts(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                dropout_rate=dropout_rate,
            )
        else:
            self.mlp = FeedForwardNetwork(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dropout_rate=dropout_rate,
            )
    
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
        Forward pass of the transformer block
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position IDs for rotary embeddings
            past_key_value: Cached key/value states for generation
            output_attentions: Whether to output attention weights
            use_cache: Whether to return key/value states for generation
            
        Returns:
            hidden_states: Output tensor
            present_key_value: Updated key/value states
            attention_weights: Attention weights (optional)
        """
        residual = hidden_states
        
        hidden_states = self.input_layernorm.forward(hidden_states)
        
        batch_size, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = np.arange(seq_len)[None, :]  # [1, seq_len]
            position_ids = np.repeat(position_ids, batch_size, axis=0)  # [batch_size, seq_len]
        
        attention_outputs = self.self_attn.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        hidden_states = attention_outputs[0]
        
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        
        hidden_states = self.mlp.forward(hidden_states)
        
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (attention_outputs[1],)
        if output_attentions:
            outputs += (attention_outputs[2],)
            
        return outputs


class LlamaMaverickModel:
    """Llama 4 Maverick model implementation in NumPy"""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 8192,
        use_moe_layers: bool = True,
        moe_intermediate_size: Optional[int] = None,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        moe_layer_freq: int = 4, 
        rms_norm_eps: float = 1e-5,
        vocab_embed_factor: Optional[float] = None,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize a Llama 4 Maverick model
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension size
            intermediate_size: Intermediate size for FFN layers
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key/value heads for group query attention
            max_position_embeddings: Maximum sequence length
            use_moe_layers: Whether to use MoE layers
            moe_intermediate_size: Intermediate size for MoE layers (if None, uses intermediate_size)
            num_experts: Number of experts in MoE layers
            num_experts_per_token: Number of experts to route each token to
            moe_layer_freq: Use MoE every nth layer
            rms_norm_eps: Epsilon for layer normalization
            vocab_embed_factor: Factor for token embeddings
            dropout_rate: Dropout probability
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.dropout_rate = dropout_rate
        
        # Initialize token embeddings
        if vocab_embed_factor is not None:
            embed_dim = int(hidden_size * vocab_embed_factor)
            self.embed_dim = embed_dim
            self.embed_in = np.random.normal(0, 0.02, (vocab_size, embed_dim))
            self.embed_proj = np.random.normal(0, 0.02, (embed_dim, hidden_size))
        else:
            self.embed_dim = hidden_size
            self.embed_in = np.random.normal(0, 0.02, (vocab_size, hidden_size))
            self.embed_proj = None
        
        # Initialize transformer blocks
        self.layers = []
        for i in range(num_hidden_layers):
            use_moe = use_moe_layers and (i % moe_layer_freq == moe_layer_freq - 1)
            layer = TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                intermediate_size=self.moe_intermediate_size if use_moe else intermediate_size,
                use_moe=use_moe,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                dropout_rate=dropout_rate,
                layer_id=i,
                max_seq_len=max_position_embeddings,
            )
            self.layers.append(layer)
        
        # Initialize final layer norm
        self.norm = LayerNorm(hidden_size, eps=rms_norm_eps)
        
        # Initialize output projection
        self.lm_head = np.random.normal(0, 0.02, (hidden_size, vocab_size))
        
    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        past_key_values: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Forward pass of the full model
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, 1, seq_len, seq_len]
            position_ids: Position IDs for rotary embeddings
            past_key_values: Cached key/value states for generation
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            use_cache: Whether to return key/value states for generation
            
        Returns:
            Dictionary with model outputs:
                - logits: Prediction logits
                - past_key_values: Updated key/value states (if use_cache)
                - hidden_states: All hidden states (if output_hidden_states)
                - attentions: All attention weights (if output_attentions)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = np.ones((batch_size, 1, seq_len, seq_len))
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = np.arange(seq_len)[None, :]  # [1, seq_len]
            position_ids = np.repeat(position_ids, batch_size, axis=0)  # [batch_size, seq_len]
        
        # Get token embeddings
        hidden_states = self.embed_in[input_ids]  # [batch_size, seq_len, embed_dim]
        
        # Project embeddings if needed
        if self.embed_proj is not None:
            hidden_states = hidden_states @ self.embed_proj  # [batch_size, seq_len, hidden_size]
        
        # Initialize past_key_values if not provided
        if past_key_values is None:
            past_key_values = [None] * self.num_hidden_layers
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_past_key_values = () if use_cache else None
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_outputs = layer.forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_past_key_values += (layer_outputs[1],)
            
            if output_attentions:
                all_attentions += (layer_outputs[2],)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        hidden_states = self.norm.forward(hidden_states)
        
        logits = hidden_states @ self.lm_head  # [batch_size, seq_len, vocab_size]
        
        outputs = {
            "logits": logits,
        }
        
        if use_cache:
            outputs["past_key_values"] = next_past_key_values
        
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
            
        if output_attentions:
            outputs["attentions"] = all_attentions
            
        return outputs


def generate(
    model: LlamaMaverickModel,
    input_ids: np.ndarray,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
):
    """
    Generate text using the model with various sampling strategies
    
    Input:
        model: The LlamaMaverickModel instance
        input_ids: Starting token IDs [batch_size, seq_len]
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Number of highest probability tokens to keep for top-k sampling
        top_p: Probability threshold for nucleus sampling
        repetition_penalty: Penalty for repeating tokens
        
    Output:
        Generated token IDs
    """
    past_key_values = None
    generated_ids = input_ids.copy()
    batch_size = input_ids.shape[0]
    
    for _ in range(max_length):
        # Forward pass with caching
        model_outputs = model.forward(
            input_ids=generated_ids[:, -1:],  # Only process the last token
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        # Get logits for next token prediction
        next_token_logits = model_outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
        past_key_values = model_outputs["past_key_values"]
        
        # Apply repetition penalty
        for b in range(batch_size):
            for token_id in set(generated_ids[b].tolist()):
                next_token_logits[b, token_id] /= repetition_penalty
                
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Apply top-k sampling
        if top_k > 0:
            indices_to_remove = np.zeros_like(next_token_logits, dtype=bool)
            for b in range(batch_size):
                topk_logits, topk_indices = np.sort(next_token_logits[b])[-top_k:], np.argsort(next_token_logits[b])[-top_k:]
                indices_to_remove[b] = ~np.isin(np.arange(next_token_logits.shape[1]), topk_indices)
            next_token_logits = np.where(indices_to_remove, -float('inf'), next_token_logits)
            
        # Apply top-p (nucleus) sampling
        if 0.0 < top_p < 1.0:
            for b in range(batch_size):
                sorted_logits = np.sort(next_token_logits[b])[::-1]
                sorted_indices = np.argsort(next_token_logits[b])[::-1]
                cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = np.zeros_like(next_token_logits[b], dtype=bool)
                indices_to_remove[sorted_indices[sorted_indices_to_remove]] = True
                next_token_logits[b] = np.where(indices_to_remove, -float('inf'), next_token_logits[b])
        
        # Sample from the filtered distribution
        probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits), axis=-1, keepdims=True)
        next_tokens = np.zeros(batch_size, dtype=np.int32)
        
        for b in range(batch_size):
            next_tokens[b] = np.random.choice(probs.shape[1], p=probs[b])
            
        # Add the next tokens to the generated sequence
        generated_ids = np.concatenate([generated_ids, next_tokens[:, None]], axis=1)
    
    return generated_ids 