import os 
from transformers import GPT2TokenizerFast
from config import config

class SimpleTokenizer:

    def __init__(self):

        if os.path.exists(config.tokenizer_path):
            print("Loading existing tokenizer...")

            #load prreviously saved model
            self.tokenizer=GPT2TokenizerFast.from_pretrained(config.tokenizer_path)
        else:
            print("creating neew tokenizer...")
            #create new tokenizer based on GPT-2
            self.tokenizer=GPT2TokenizerFast.from_pretrained('gpt2')

            #Add special tokens that our model needs
            special_tokens = {
                "pad_token": "<PAD>",      # For padding shorter sequences
                "eos_token": "<EOS>",      # End of sequence marker
                "bos_token": "<BOS>",      # Beginning of sequence marker
                "unk_token": "<UNK>"       # hanales unknown words
            }

            #add these tokens to the the tokenizers vocublary
            self.tokenizer.add_special_tokens(special_tokens)

            #create directory if it doesnt exist
            os.makedirs(os.path.dirname(config.tokenizer_path), exist_ok=True)

            #save the tokenizer for future use
            self.tokenizer.save_pretrained(config.tokenizer_path)

        #store vocublary size for model configuration
        self.vocab_size= len(self.tokenizer)
        print(f"Tokenizer ready with vocublary size: {self.vocab_size}")


    def encode(self,text):
        """
        Convert text string into list of token IDs (numbers)
        
        Args:
            text (str): Raw text to convert
            
        Returns:
            list: List of integers representing the tokens
        """
        # Add beginnning of sequence token and convert to numbers
        text_with_bos="<BOS>" + text

        #covert text to token IDs
        token_ids=self.tokenizer.encode(text_with_bos,
                                        add_special_tokens=False,
                                        max_length=config.max_length,
                                        truncation= True)
        
        return token_ids
    
    def full_encode(self, text):
        """No truncation — for entire corpus"""
        text_with_bos = "<BOS>" + text
        token_ids = self.tokenizer.encode(
            text_with_bos,
            add_special_tokens=False,
            truncation=False  # <<< THIS FIXES YOUR DATASET ISSUE
        )
        return token_ids

    def decode(self,token_ids):

        text=self.tokenizer.decode(token_ids,skip_special_tokens=True)

        # Clean up the output
        text=text.strip()

        return text
    
    def encode_batch(self,texts):
        # Add BOS token to all texts
        texts_with_bos = ["<BOS>" + text for text in texts]
        
        # Tokenize all texts together
        encoded = self.tokenizer(texts_with_bos,
                               padding=True,  # Pad shorter sequences
                               truncation=True,  # Cut longer sequences
                               max_length=config.max_length,
                               return_tensors="pt")  # Return PyTorch tensors
        
        return encoded
    
    def get_vocab_size(self):
        """Return the size of the vocabulary"""
        return self.vocab_size
    
    def get_special_token_ids(self):
        """Return the IDs of special tokens for easy access"""
        return {
            'pad_id': self.tokenizer.pad_token_id,
            'eos_id': self.tokenizer.eos_token_id,
            'bos_id': self.tokenizer.bos_token_id,
            'unk_id': self.tokenizer.unk_token_id
        }