import re
import numpy as np
from typing import List, Dict, Tuple, Optional

class SimpleTokenizer:
    """
    Simple tokenizer with expanded vocabulary for transformer models
    """
    
    def __init__(self, vocab_size: int = 32000):
        """Initialize tokenizer with larger vocabulary for transformer compatibility"""
        self.vocab_size = vocab_size
        
        # Special token IDs - reserve first few tokens
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
        special_tokens = {
            "<pad>": self.pad_token_id,
            "<bos>": self.bos_token_id,
            "<eos>": self.eos_token_id,
            "<unk>": self.unk_token_id,
        }
        
        # Start character mapping after special tokens
        start_char_id = max(special_tokens.values()) + 1
        max_char_id = min(vocab_size - 1, start_char_id + 255)  # Leave room for special tokens
        
        # Character-level vocab - map ASCII chars to IDs after special tokens
        self.char_to_id = {chr(i): start_char_id + i for i in range(min(256, max_char_id - start_char_id + 1))}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        # Add special tokens to vocabulary
        self.char_to_id.update(special_tokens)
        self.id_to_char.update({v: k for k, v in special_tokens.items()})
    
    def encode(self, text: str) -> List[int]:
        """
        Encode a string into token IDs
        
        Input:
            text: Input text string
            
        Output:
            List of token IDs
        """
        tokens = []
        
        # Add BOS token
        tokens.append(self.bos_token_id)
        
        # Character-level tokenization
        for char in text:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                tokens.append(self.unk_token_id)
        
        # Add EOS token
        tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to a string
        
        Input:
            token_ids: List of token IDs
            
        Output:
            Decoded text string
        """
        text = ""
        
        for token_id in token_ids:
            # Skip special tokens
            if token_id in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                continue
            
            # Ensure token_id is within vocabulary range
            if token_id >= self.vocab_size:
                token_id = self.unk_token_id
                
            # Convert token to character
            if token_id in self.id_to_char:
                text += self.id_to_char[token_id]
            else:
                text += self.id_to_char[self.unk_token_id]
        
        return text
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch of texts into token IDs
        
        Inputs:
            texts: List of input text strings
            max_length: Maximum sequence length (truncate if longer)
            padding: Whether to pad sequences to the same length
            truncation: Whether to truncate sequences longer than max_length
            
        Output:
            NumPy array of token IDs with shape [batch_size, seq_len]
        """
        encoded_texts = [self.encode(text) for text in texts]
        
        # Determine sequence length
        if max_length is None and padding:
            max_length = max(len(tokens) for tokens in encoded_texts)
        elif max_length is None:
            max_length = max(len(tokens) for tokens in encoded_texts)
        
        # Truncate if needed
        if truncation:
            encoded_texts = [tokens[:max_length] for tokens in encoded_texts]
        
        # Pad if needed
        if padding:
            encoded_texts = [
                tokens + [self.pad_token_id] * (max_length - len(tokens))
                for tokens in encoded_texts
            ]
        
        return np.array(encoded_texts)


def text_generation_demo(
    model,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
):
    """
    Demonstration of text generation with the model
    
    Args:
        model: The LlamaMaverickModel instance
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Text prompt to start generation
        max_length: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Number of highest probability tokens for top-k sampling
        top_p: Probability threshold for nucleus sampling
        repetition_penalty: Penalty for repeating tokens
        
    Returns:
        Generated text and debug info
    """
    from transformer import generate
    
    input_ids = tokenizer.batch_encode([prompt])
    
    generated_ids = generate(
        model=model,
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    
    new_tokens = generated_ids[0, input_ids.shape[1]:]
    full_text = tokenizer.decode(generated_ids[0].tolist())
    generated_text = tokenizer.decode(new_tokens.tolist())
    
    # Add debug information
    token_info = [
        f"{i}:{t}" for i, t in enumerate(generated_ids[0].tolist())
    ]
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": full_text,
        "token_ids": token_info,  # Debug info showing token IDs
    } 