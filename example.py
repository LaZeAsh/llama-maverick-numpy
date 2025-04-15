import numpy as np
from transformer import LlamaMaverickModel, generate
from tokenization import SimpleTokenizer, text_generation_demo

def main():
    print("Initializing Llama 4 Maverick model...")
    # Use vocab_size of at least 256 (for ASCII) + 4 (for special tokens)
    vocab_size = 260
    model = LlamaMaverickModel(
        vocab_size=vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        use_moe_layers=True,
        num_experts=4,
        num_experts_per_token=2,
        moe_layer_freq=4,
        dropout_rate=0.0,
        rms_norm_eps=1e-5
    )
    
    # Create a simple tokenizer with matching vocab size
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    
    # Standard forward pass example
    input_text = "Hello, world!"
    input_ids = tokenizer.batch_encode([input_text])
    
    print(f"Input text: {input_text}")
    print("Input shape:", input_ids.shape)
    
    # Forward pass through the model
    outputs = model.forward(input_ids=input_ids)
    
    # Get the logits
    logits = outputs["logits"]
    print("Output logits shape:", logits.shape)
    
    # Generate some text
    # print("\nText generation example:")
    prompt = "Once upon a time, there was a"
    
    generation_result = text_generation_demo(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=30,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
    )
    
    # print(f"Prompt: {generation_result['prompt']}")
    print(f"Generated text: {generation_result['generated_text']}")
    # print(f"Full text: {generation_result['full_text']}")
    


if __name__ == "__main__":
    main() 