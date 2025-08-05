import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from config import config
from tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()

class TextDataset(Dataset):
    
    def __init__(self, data_path):
        
        self.sequences = []
        self.load_data(data_path)

    def load_data(self,data_path):

        print("Loading and tokenizing data ....")

        text_files = glob.glob(os.path.join(data_path,"*.txt"))

        if not text_files:
            print("No text files found")
            self.create_sample_data(data_path)
            text_files=[os.path.join(data_path, "sample.txt")]

        all_text = ""

        for file_path in text_files:
            print(f"Loading: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                file_text = f.read()
                all_text +=file_text + "\n"

        print(f"Total text length: {len(all_text)} characters")

        self.create_sequences(all_text)

        print(f"Created {len(self.sequences)} training sequences")

    def create_sample_data(self, data_path):
        """
        Create sample training data if no files are provided.
        This gives users something to test with immediately.
        """
        sample_text = """
        The quick brown fox jumps over the lazy dog. This is a sample text for training our language model.
        Artificial intelligence is transforming the world. Machine learning models can understand and generate human language.
        Deep learning uses neural networks with many layers. Transformers are a type of neural network architecture.
        Natural language processing helps computers understand text. Language models can write stories, answer questions, and more.
        Python is a popular programming language for AI. PyTorch is a framework for building neural networks.
        Training language models requires lots of text data. The model learns patterns from examples.
        Attention mechanisms help models focus on relevant information. Self-attention is key to transformer success.
        Embeddings convert words into numerical representations. Position encodings help models understand word order.
        Backpropagation trains neural networks by computing gradients. Optimization algorithms update model parameters.
        Generative AI can create new content from learned patterns. Large language models have billions of parameters.
        """
        
        # Create data directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        
        # Write sample data to file
        sample_file = os.path.join(data_path, "sample.txt")
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_text.strip())
        
        print(f"Created sample data file: {sample_file}")

    
    def create_sequences(self, text):
        """
        Convert raw text into training sequences of fixed length.
        
        Args:
            text (str): Raw text to convert
        """
        # Tokenize the entire text
        token_ids = tokenizer.full_encode(text)
        print(f"Token count: {len(token_ids)}")
        # Create overlapping sequences for training
        sequence_length = config.max_length
        
        # Generate sequences with sliding window
        for i in range(0, len(token_ids) - sequence_length, sequence_length // 2):
            # Extract sequence of tokens
            sequence = token_ids[i:i + sequence_length]
            
            # Only keep sequences that are full length
            if len(sequence) == sequence_length:
                self.sequences.append(torch.tensor(sequence, dtype=torch.long))
        
        # If we don't have enough data, repeat sequences
        if len(self.sequences) < config.batch_size * 10:
            print("Warning: Limited training data. Repeating sequences...")
            original_sequences = self.sequences.copy()
            while len(self.sequences) < config.batch_size * 20:
                self.sequences.extend(original_sequences)
    
    def __len__(self):
        """Return the number of sequences in the dataset"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Args:
            idx (int): Index of the sequence to return
            
        Returns:
            torch.Tensor: Token sequence for training
        """
        return self.sequences[idx]

def create_data_loader(data_path, batch_size=None, shuffle=True):
    """
    Create PyTorch DataLoader for training.
    DataLoader handles batching and shuffling automatically.
    
    Args:
        data_path (str): Path to data directory
        batch_size (int): Number of sequences per batch
        shuffle (bool): Whether to shuffle data between epochs
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    if batch_size is None:
        batch_size = config.batch_size
    
    # Create dataset
    dataset = TextDataset(data_path)
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return data_loader

def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Handles batching of variable-length sequences.
    
    Args:
        batch (list): List of tensor sequences
        
    Returns:
        torch.Tensor: Batched sequences
    """
    # Stack all sequences into a batch
    batch_tensor = torch.stack(batch)
    return batch_tensor

# Utility function to test data loading
def test_data_loading():
    """Test function to verify data loading works correctly"""
    print("Testing data loading...")
    
    # Create data loader
    data_loader = create_data_loader(config.data_path)
    
    # Get one batch
    for batch in data_loader:
        print(f"Batch shape: {batch.shape}")
        print(f"First sequence tokens: {batch[0][:10]}")
        
        # Decode first sequence to verify it makes sense
        decoded = tokenizer.decode(batch[0].tolist())
        print(f"Decoded text: {decoded[:100]}...")
        break
    
    print("Data loading test completed!")

if __name__ == "__main__":
    test_data_loading()