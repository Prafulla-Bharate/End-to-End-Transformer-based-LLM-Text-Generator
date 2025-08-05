import torch 
import torch.nn as nn
import torch.nn.functional as F
from config import config
import math

class PositionalEncoding(nn.Module):

    def __init__(self,d_model,max_length):
        super().__init__()

        #create a matrix to hold positional encodeings
        pe=torch.zeros(max_length,d_model)

        #create positions indices[0,1,2,3,....,max_length-1]
        position= torch.arange(0,max_length).unsqueeze(1).float()

        #create division term for sine/cosine functions
        div_term = torch.exp(torch.arange(0,d_model,2).float()*
                             -( math.log(10000.0)/d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even positions get sine
        pe[:, 1::2] = torch.cos(position * div_term)# Odd positions get cosine

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    
    def forward(self, x):
        """Add positional encoding to input embeddings"""
        seq_len = x.size(1)  # Get sequence length
        # Add positional encoding (only up to sequence length)
        return x + self.pe[:, :seq_len]
    


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model=d_model
        self.n_heads=n_heads
        self.d_k = d_model // n_heads

    #Linear projections for quries , keys, and values
        self.w_q= nn.Linear(d_model,d_model)  #query projection
        self.w_k = nn.Linear(d_model, d_model)  # Key projection  
        self.w_v = nn.Linear(d_model, d_model)  # Value projection
        self.w_o = nn.Linear(d_model, d_model)  # Output projection

        #Dropout for regularization
        self.dropout= nn.Dropout(config.dropout)


    def forward(self, x, mask=None):
        batch_size, seq_len,d_model=x.shape

        #Generate queries ,keys and values
        Q = self.w_q(x)
        K=self.w_k(x)
        V=self.w_v(x)

        #Reshape for multihead attention
        Q=Q.view(batch_size,seq_len,self.n_heads,self.d_k).transpose(1,2)
        K=K.view(batch_size,seq_len,self.n_heads,self.d_k).transpose(1,2)
        V=V.view(batch_size,seq_len,self.n_heads,self.d_k).transpose(1,2)

        #compute attention scores
        scores= torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)

        #Apply causal mask(prevent looking at future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # apply Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights= self.dropout(attention_weights)

        #Apply attention values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads and project output
        context= context.transpose(1,2).contiguous().view(
            batch_size, seq_len, d_model
        )

        output = self.w_o(context)
        return output
    

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Applies the same transformation to each position independently.
    """
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Two linear transformations with ReLU activation
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # First linear layer + ReLU + dropout
        x = self.dropout(F.relu(self.linear1(x)))
        # Second linear layer
        x = self.linear2(x)
        return x
    

class TransformerBlock(nn.Module):
    """
    A single transformer decoder block.
    Contains multi-head attention and feed-forward network with residual connections.
    """
    
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        # Multi-head self-attention
        self.attention = MultiHeadAttention(d_model, n_heads)
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff)
        # Layer normalization (applied before each sub-layer)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # Dropout for residual connections
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x, mask=None):
        # Pre-norm architecture: LayerNorm before attention
        attn_output = self.attention(self.ln1(x), mask)
        # Residual connection + dropout
        x = x + self.dropout(attn_output)
        
        # Pre-norm architecture: LayerNorm before feed-forward
        ff_output = self.feed_forward(self.ln2(x))
        # Residual connection + dropout
        x = x + self.dropout(ff_output)
        
        return x 
    

class GPTModel(nn.Module):
    """
    Complete GPT model implementation.
    Combines embeddings, positional encoding, and transformer blocks.
    """
    
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Token embeddings: convert token IDs to dense vectors
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        
        # Positional encoding: add position information
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_length)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_layers)
        ])
        
        # Final layer normalization
        self.ln_final = nn.LayerNorm(config.d_model)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights using Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len):
        """Create lower triangular mask to prevent attention to future tokens"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final layer normalization
        x = self.ln_final(x)
        
        # Project to vocabulary size
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        # Calculate loss if targets provided (training mode)
        loss = None
        if targets is not None:
            # Shift targets: predict next token for each position
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            # Flatten for cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    def generate(self, input_ids, max_length, temperature=0.8, top_k=50):
        """
        Generate text using the trained model.
        Uses top-k sampling for more interesting outputs.
        """
        self.eval()  # Set to evaluation mode
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for next token
                logits, _ = self.forward(generated)
                
                # Get logits for last position (next token prediction)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    # Get top-k values and indices
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    
                    # Create mask for top-k tokens
                    mask = torch.full_like(next_token_logits, float('-inf'))
                    mask.scatter_(1, top_k_indices, top_k_values)
                    next_token_logits = mask
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we exceed max length
                if generated.size(1) >= config.max_length:
                    break
        
        return generated
