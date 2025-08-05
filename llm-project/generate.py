import torch
import torch.nn.functional as F
from config import config
from model import GPTModel
import argparse
import os
from tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()

class TextGenerator:

    def __init__(self, model_path=None):
        self.device = torch.device(config.device)
        print(f"Using device: {self.device}")

        if model_path is None:
            model_path = config.model_save_path

        # 🔧 Assign the model returned from load_model() to self.model
        self.model = self.load_model(model_path)

        self.model.eval()
        print("TextGenerator ready!")


    def load_model(self, model_path):
        """
        Load trained model weights.
        
        Args:
            model_path (str): Path to model checkpoint
        """
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load model weights (handle both full checkpoint and state dict)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
                print("Loaded model state dict")
                
        else:
            print(f"Warning: Model file not found at {model_path}")
            print("Using randomly initialized model (will generate random text)")


    def generate_text(self, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9):
        """
        Generate text using the trained model with various sampling strategies.
        
        Args:
            prompt (str): Starting text to generate from
            max_length (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0.1=conservative, 1.5=creative)
            top_k (int): Only consider top k most likely tokens (0=disabled)
            top_p (float): Nucleus sampling threshold (0.9=conservative, 0.95=creative)
            
        Returns:
            str: Generated text
        """
        # Encode the prompt
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        print(f"Generating text with prompt: '{prompt}'")
        print(f"Parameters: max_length={max_length}, temperature={temperature}, top_k={top_k}, top_p={top_p}")
        print("-" * 50)
        
        generated_ids = input_tensor.clone()
        
        # Generate tokens one by one
        with torch.no_grad():
            for step in range(max_length):
                # Get model predictions
                logits, _ = self.model(generated_ids)
                
                # Get logits for next token (last position)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    # Get top-k values and indices
                    top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    # Set all other values to negative infinity
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_values)
                
                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to generated sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit max sequence length
                if generated_ids.size(1) >= config.max_length:
                    break
                
                # Optional: Stop if we generate end-of-sequence token
                special_tokens = tokenizer.get_special_token_ids()
                if next_token.item() == special_tokens.get('eos_id'):
                    break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
        
        return generated_text
    
    def interactive_generation(self):
        """Interactive text generation session"""
        print("Interactive Text Generation")
        print("Enter prompts to generate text. Type 'quit' to exit.")
        print("Type 'help' for options.")
        print("=" * 50)
        
        # Default generation parameters
        max_length = 100
        temperature = 0.8
        top_k = 50
        top_p = 0.9
        
        while True:
            try:
                # Get user input
                user_input = input("\nEnter prompt (or command): ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  quit - Exit the program")
                    print("  help - Show this help message")
                    print("  set temp <value> - Set temperature (0.1-2.0)")
                    print("  set length <value> - Set max generation length")
                    print("  set topk <value> - Set top-k value")
                    print("  set topp <value> - Set top-p value")
                    print("  show settings - Show current generation settings")
                    continue
                
                elif user_input.lower().startswith('set '):
                    # Handle parameter changes
                    parts = user_input.split()
                    if len(parts) == 3:
                        param, value = parts[1], parts[2]
                        try:
                            if param == 'temp':
                                temperature = float(value)
                                print(f"Temperature set to {temperature}")
                            elif param == 'length':
                                max_length = int(value)
                                print(f"Max length set to {max_length}")
                            elif param == 'topk':
                                top_k = int(value)
                                print(f"Top-k set to {top_k}")
                            elif param == 'topp':
                                top_p = float(value)
                                print(f"Top-p set to {top_p}")
                            else:
                                print(f"Unknown parameter: {param}")
                        except ValueError:
                            print(f"Invalid value for {param}: {value}")
                    continue
                
                elif user_input.lower() == 'show settings':
                    print(f"\nCurrent settings:")
                    print(f"  Temperature: {temperature}")
                    print(f"  Max length: {max_length}")
                    print(f"  Top-k: {top_k}")
                    print(f"  Top-p: {top_p}")
                    continue
                
                elif not user_input:
                    print("Please enter a prompt or command.")
                    continue
                
                # Generate text
                generated_text = self.generate_text(
                    prompt=user_input,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                print(f"\nGenerated text:\n{generated_text}")
                
            except KeyboardInterrupt:
                print("\nGeneration interrupted.")
                continue
            except Exception as e:
                print(f"Error during generation: {e}")
                continue
    
    def batch_generation(self, prompts, **generation_kwargs):
        """
        Generate text for multiple prompts.
        
        Args:
            prompts (list): List of prompt strings
            **generation_kwargs: Generation parameters
            
        Returns:
            list: List of generated texts
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Generating {i+1}/{len(prompts)}: {prompt[:50]}...")
            generated_text = self.generate_text(prompt, **generation_kwargs)
            results.append(generated_text)
        
        return results
    
    def creative_writing_prompts(self):
        """Generate text for a set of creative writing prompts"""
        prompts = [
            "Once upon a time in a world where",
            "The last person on Earth sat alone in a room. There was a knock on the door.",
            "In the year 2050, artificial intelligence",
            "The secret to happiness is",
            "If I could travel back in time, I would",
            "The most important lesson I've learned is",
            "In a parallel universe where gravity works backwards",
            "The old lighthouse keeper had a secret"
        ]
        
        print("Generating creative writing samples...")
        print("=" * 50)
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 30)
            
            generated_text = self.generate_text(
                prompt=prompt,
                max_length=80,
                temperature=0.9,  # More creative
                top_k=40,
                top_p=0.85
            )
            
            print(generated_text)
            print()

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Generate text using trained GPT model")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--interactive", action="store_true", help="Start interactive session")
    parser.add_argument("--creative", action="store_true", help="Generate creative writing samples")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TextGenerator(model_path=args.model)
    
    if args.interactive:
        # Interactive mode
        generator.interactive_generation()
    
    elif args.creative:
        # Creative writing mode
        generator.creative_writing_prompts()
    
    elif args.prompt:
        # Single prompt generation
        generated_text = generator.generate_text(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        print("Generated text:")
        print("=" * 50)
        print(generated_text)
    
    else:
        # Default: interactive mode
        print("No specific mode selected. Starting interactive generation...")
        generator.interactive_generation()

if __name__ == "__main__":
    main()
