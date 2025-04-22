# Llama 4 Maverick NumPy Implementation

A lightweight implementation of the Llama 4 Maverick architecture in pure NumPy. This repository provides a decoder-only transformer similar to the Llama 4 Maverick model, focusing on its key architectural features.

## Features

- **Group Query Attention (GQA)**: Efficient attention mechanism where multiple query heads share the same key/value heads.
- **Rotary Position Embeddings (RoPE)**: Position encoding that captures relative positions between tokens.
- **Mixture of Experts (MoE)**: Selective routing of tokens through specialized expert networks.
- **SwiGLU Activation**: State-of-the-art activation function for feed-forward networks.
- **Decoder-only Architecture**: Causal transformer model suitable for text generation tasks.

## Files

- `attention.py`: Implementation of Group Query Attention and RoPE.
- `ffn.py`: Feed-forward network and Mixture of Experts implementation.
- `transformer.py`: Transformer block and full model architecture.
- `example.py`: Usage example demonstrating model inference and generation.

## Usage

```python
import numpy as np
from transformer import LlamaMaverickModel, generate

# Create a model
model = LlamaMaverickModel(
    vocab_size=32000,
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    use_moe_layers=True,
    num_experts=4,
    num_experts_per_token=2,
)

# Generate text
input_ids = np.array([[1, 2, 3, 4, 5]])  # Initial token sequence
generated_ids = generate(
    model=model,
    input_ids=input_ids,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)
```

## Key Components

### Group Query Attention (GQA)

Group Query Attention reduces the computational cost of attention by having multiple query heads share the same key/value heads. This approach sits between Multi-head Attention (separate K/V per head) and Multi-query Attention (single K/V for all heads).

### Mixture of Experts (MoE)

The MoE layer routes each token to a subset of specialized feed-forward networks (experts) based on the token's content. This allows the model to develop specialized processing pathways for different types of inputs, effectively increasing model capacity without a proportional increase in computation.

### Running the Example

```bash
python example.py
```

## Limitations

This implementation is for educational purposes and focuses on architectural clarity rather than performance optimization. For production use cases, consider frameworks like PyTorch or TensorFlow with proper GPU acceleration.

## Architectural Comparison



## References

- [Llama 4 Multimodal Intelligence](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Group Query Attention](https://arxiv.org/abs/2305.13245)
- [Early Fusion](https://arxiv.org/abs/2405.09818)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Rotary Positional Embeddings](https://arxiv.org/abs/2104.09864)
- [Scalable Softmax](https://arxiv.org/pdf/2501.19399)
- [Hybrid Attention Strategy](https://arxiv.org/html/2501.18795v1)

<!-- - [Llama 2 Technical Report](https://arxiv.org/abs/2307.09288)
- [Mixture of Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) -->
