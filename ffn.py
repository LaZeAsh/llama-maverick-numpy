import numpy as np
from typing import Optional, List, Tuple

class FeedForwardNetwork:
    """Standard Feed-Forward Network for Transformer models"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        dropout_rate: float = 0.0,
    ):
        """
        Initialize a Feed-Forward Network
        
        Args:
            hidden_size: Input and output dimension
            intermediate_size: Dimension of the intermediate layer
            hidden_act: Activation function ("gelu", "relu", "silu")
            dropout_rate: Dropout probability
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout_rate = dropout_rate
        
        # Initialize weights
        self.gate_proj = np.random.normal(0, 0.02, (hidden_size, intermediate_size))
        self.up_proj = np.random.normal(0, 0.02, (hidden_size, intermediate_size))
        self.down_proj = np.random.normal(0, 0.02, (intermediate_size, hidden_size))
        
    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.hidden_act == "gelu":
            # GELU approximation
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        elif self.hidden_act == "relu":
            # Rectified Linear Activation Unit
            return np.maximum(0, x)
        elif self.hidden_act == "silu":
            # SiLU (Swish) activation: x * sigmoid(x)
            return x * (1 / (1 + np.exp(-x)))
        else:
            raise ValueError(f"Unsupported activation function: {self.hidden_act}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the FFN
        
        Input:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Output:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # SwiGLU activation as used in Llama models
        gate_output = self._activation_fn(x @ self.gate_proj)  # [batch_size, seq_len, intermediate_size]
        up_output = x @ self.up_proj  # [batch_size, seq_len, intermediate_size]
        
        # Element-wise multiply gate and up projections
        intermediate_output = gate_output * up_output  # [batch_size, seq_len, intermediate_size]
        
        # Down projection
        output = intermediate_output @ self.down_proj  # [batch_size, seq_len, hidden_size]
        
        # Apply dropout during training
        if self.dropout_rate > 0 and hasattr(self, 'training') and self.training:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, output.shape) / (1 - self.dropout_rate)
            output = output * dropout_mask
            
        return output


class MixtureOfExperts:
    """Mixture of Experts layer for Llama 4 Maverick"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        hidden_act: str = "silu",
        dropout_rate: float = 0.0,
    ):
        """
        Initialize a Mixture of Experts layer
        
        Args:
            hidden_size: Input and output dimension
            intermediate_size: Dimension of the intermediate layer in each expert
            num_experts: Total number of experts
            num_experts_per_token: Number of experts to route each token to
            hidden_act: Activation function for the experts
            dropout_rate: Dropout probability
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        assert num_experts_per_token <= num_experts, "num_experts_per_token must be <= num_experts"
        
        # Create experts (each expert is an FFN)
        self.experts = [
            FeedForwardNetwork(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                dropout_rate=dropout_rate
            )
            for _ in range(num_experts)
        ]
        
        # Router parameters
        self.router = np.random.normal(0, 0.02, (hidden_size, num_experts))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the MoE
        
        Input:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Output:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape input for routing
        x_reshaped = x.reshape(-1, self.hidden_size)  # [batch_size * seq_len, hidden_size]
        
        # Compute routing logits
        router_logits = x_reshaped @ self.router  # [batch_size * seq_len, num_experts]
        
        # Find top-k experts per token
        expert_indices = np.argsort(-router_logits, axis=1)[:, :self.num_experts_per_token]  # [batch_size * seq_len, num_experts_per_token]
        
        # Compute router probabilities (softmax over selected experts)
        routing_weights = np.zeros((batch_size * seq_len, self.num_experts))
        for i in range(batch_size * seq_len):
            selected_logits = router_logits[i, expert_indices[i]]
            selected_weights = np.exp(selected_logits - np.max(selected_logits))
            selected_weights = selected_weights / np.sum(selected_weights)
            routing_weights[i, expert_indices[i]] = selected_weights
        
        # Initialize output tensor
        final_output = np.zeros((batch_size * seq_len, self.hidden_size))
        
        # Process tokens through their assigned experts
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = routing_weights[:, expert_idx] > 0
            if not np.any(expert_mask):
                continue
                
            # Select tokens for this expert
            expert_inputs = x_reshaped[expert_mask]
            
            # Get expert outputs
            expert_output = self.experts[expert_idx].forward(expert_inputs)
            
            # Scale outputs by routing weights
            expert_weights = routing_weights[expert_mask, expert_idx][:, None]
            scaled_expert_output = expert_output * expert_weights
            
            # Add to final output
            final_output[expert_mask] += scaled_expert_output
            
        # Reshape back to original dimensions
        output = final_output.reshape(batch_size, seq_len, self.hidden_size)
        
        return output
