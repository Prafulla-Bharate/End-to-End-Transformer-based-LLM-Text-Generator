import torch

class ModelConfig:

    vocab_size= 7000
    max_length= 256
    n_layers = 2
    n_heads = 4
    d_model= 256
    d_ff = 1024
    dropout = 0.1

    #training parameters
    batch_size= 8
    learning_rate= 3e-4
    num_epochs= 10
    warmup_steps=100

    #Generation parameters
    max_generate_length= 100
    temprature= 0.8
    top_k= 50

    #file path
    data_path="llm-project/data"
    model_save_path="saved_models/gpt_model.pt"
    tokenizer_path="saved_models/tokenizer"

    #Device Configuration
    device= "cuda" if torch.cuda.is_available() else "cpu"

# create a global instance of config that other modules can import
config=ModelConfig()