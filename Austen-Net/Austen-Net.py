# Project Title: Infinite Text Generation in the Style of Different Authors
# Methodology: Custom Decoder Transformer Deep Neural Network from Scratch
# Author: Griffin Dean Kent
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt



##############################################################################################################################################################
############################################################# Decoder Transformer Language Model #############################################################
##############################################################################################################################################################

###############################################################################
############################# Positional Encoding #############################
###############################################################################
class PositionalEncoding(nn.Module):
    """
    This is a standard module to add sinusoidal positional encodings to the initial
    token embeddings to help the model incorporate positional information about the tokens.
    Pytorch's documentation uses a very similar implementation
    see https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch#77445896
    """
    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        #Create a matrix "pos_encodings" of shape (max_len, model_dim) that holds the positional encodings
        pos_encodings = torch.zeros(max_len, model_dim)
        # Make a vector of positional values from (0, 1, ..., max_len - 1). Then use .unsqueeze to turn it into a column vector
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the div_term (exponential decay for each dimension)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        # Apply sine to even indices and cosine to odd indices
        pos_encodings[:, 0::2] = torch.sin(position * div_term)
        pos_encodings[:, 1::2] = torch.cos(position * div_term)
        # Add an extra batch dimension (1, max_len, d_model) so it can be added to token embeddings
        pos_encodings = pos_encodings.unsqueeze(0)
        self.register_buffer('pos_encodings', pos_encodings)
    
    def forward(self, x):
        x = x + self.pos_encodings[:, :x.size(1), :]
        return x
        

###############################################################################
##################### "Look-Ahead" Masking Helper Function ####################
###############################################################################
def generate_look_ahead_mask(size):
    """
    This is a helper function that will return a triangular matrix of dimensions size x size which
    will prevent positions from attending to subsequent positions.
    """
    #This is a triangular matrix with "True" below the diagonal and "False" above it
    mask = torch.tril(torch.ones(size, size), diagonal=0).bool() 
    return mask

###############################################################################
############################### Attention Block ###############################
###############################################################################
class AttentionHead(nn.Module):
    """
    This is the definition of a single-headed attention block. Note the following definitions.
    batch: The batch size, indicating the number of sequences (i.e., sentences)
           that are being proccessed simultaneously
    heads: The number of attention heads in the multi-headed attention block
    seq_len: The length of the input sequence (i.e., the number of words or tokens)
             in each example. Each token (or word) in the sequence will have its own
             vector for each head
    dim_K: The dimensionality of the key and query vectors
    
    Parameters
    ----------
        head_dim: This is the dimension of the Key and Query vectors that the input token embedding is mapped to
        mask: If not None, then this is a matrix of boolean values that indicate which elements in the attention scores to mask
        dropout: This is the percentage of model weights to randomly not update
    """
    def __init__(self, model_dim, head_dim, dropout=0.1):
        super().__init__()
        self.head_dim = head_dim
        #Define the matrices of learnable weights
        self.key = nn.Linear(model_dim, head_dim)
        self.query = nn.Linear(model_dim, head_dim)
        self.value = nn.Linear(model_dim, head_dim)
        #Define the dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        """
        Parameters
        ----------
        x : The input tensor representing a batch of embeddings. It has dimensions (batch_size, seq_len, d_model)

        Returns
        -------
        attn : The resulting attended embedding vector
        """
        #Compute Key, Query, and Value embeddings for the input x
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        #Compute Similarity Scores
        KT = K.transpose(-2, -1) #This swaps the dimensions in index positions -1 and -2 (i.e., seq_len and dim_K)
        G = (Q @ KT) / (self.head_dim**(0.5)) #This results in a seq_len x seq_len matrix of attention scores that measure the similarity between every token pair
        #Check to se if masking is performed. This will implicitly add the masking matrix M (mask) to the raw attention scores to prevent tokens from communicating with tokens in the future
        if mask is not None:
            G = G.masked_fill(mask==False, float('-inf')) #This replaces the entries in the mask matrix with values of -inf
        #Apply softmax
        G = F.softmax(G, dim=-1)
        G = self.dropout(G)
        #Multiply the probabilities by the value tensor V
        attn = G @ V
        return attn
    
###############################################################################
######################### Multi-Headed Attention Block ########################
###############################################################################
class MultiHeadAttention(nn.Module):
    """
    This is the definition of a multi-headed attention block. Note the following definitions.
    batch: The batch size, indicating the number of sequences (i.e., sentences)
           that are being proccessed simultaneously
    heads: The number of attention heads in the multi-headed attention block
    seq_len: The length of the input sequence (i.e., the number of words or tokens)
             in each example. Each token (or word) in the sequence will have its own
             vector for each head
    dim_K: The dimensionality of the key and query vectors
    
    Parameters
    ----------
        model_dim: This is the overal dimension of the model and is the size of the initial token embeddings
        num_heads: This is the number of attention blocks that are used
        mask: If not None, then this is a matrix of boolean values that indicate which elements in the attention scores to mask
        dropout: This is the percentage of model weights to randomly not update
    """
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.head_dim = model_dim // num_heads # head_dim is the dimension of the key and query vectors that the input token embedding is mapped to
        #Define the multiple attention blocks
        self.heads = nn.ModuleList([AttentionHead(model_dim, self.head_dim, dropout) for _ in range(num_heads)])
        #Define the linear output layer that combines the embeddings from the different attention blocks
        self.out_linear = nn.Linear(model_dim, model_dim)
        #Define the dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        out = torch.cat([attn(x, mask) for attn in self.heads], dim=-1) #Concatenate all of the attention embeddings
        out = self.out_linear(out) #Project the vector back into the embedding space
        out = self.dropout(out)
        return out
        
###############################################################################
########################### Feed-Forward (MLP) Block ##########################
###############################################################################
class MLP(nn.Module):
    """
    This is a simple feed-forward MLP network consisting of two linear layers with a ReLU activation in between for nonlinearity
    
    Parameters
    ----------
        model_dim: This is the overal dimension of the model and is the size of the initial token embeddings
        hidden_dim: This is the dimension of the hidden layer in the MLP
        dropout: This is the percentage of model weights to randomly not update
    """
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.feedforward = nn.Sequential(nn.Linear(model_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, model_dim),
                                         nn.Dropout(dropout))
        
    def forward(self, x):
        return self.feedforward(x)
        
###############################################################################
###################### Transformer Decoder Layer (Block) ######################
###############################################################################
class TransformerDecoderLayer(nn.Module):
    """
    This is a single decoder block, which contains the following:
        - A multi-headed self-attention (when masking) layer to allow the tokens to communicate with each other
        - A fully-connected feed-forward layer to identify nonlinear patterns
    Both of these layers are preceded by layer normalizations
    
    Parameters
    ----------
        model_dim: This is the overal dimension of the model and is the size of the initial token embeddings
        num_heads: This is the number of attention blocks that are used
        hidden_dim: This is the dimension of the hidden layer in the MLP
        mask: If not None, then this is a matrix of boolean values that indicate which elements in the attention scores to mask
        dropout: This is the percentage of model weights to randomly not update
    """
    def __init__(self, model_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.multi_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.mlp = MLP(model_dim, hidden_dim, dropout)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
    
    def forward(self, x, mask):
        #The "x + ..." terms here define the residual (or "skip") connections
        x = x + self.multi_attn(self.ln1(x), mask) #Multi-headed attention layer after the First normalization layer
        x = x + self.mlp(self.ln2(x)) #Feed-forward layer after the Second normalization layer
        return x
        
###############################################################################
################### Full Transformer Decoder Language Model ###################
###############################################################################
class TransformerLanguageModel(nn.Module):
    """
    This is a Decoder Transformer Language Model. It performs the following functionality in sequential order:
        - An initial embedding layer that converts tokens to vectors
        - A positional encoding layer to incorporate token order information
        - A stack of decoder blocks
        - A final MLP that converts the learned embeddings into next-token predictions in the vocabulary
    
    Parameters
    ----------
        vocab_size: This is the number of unique tokens that are present in the dataset and are possible to predict
        model_dim: This is the overal dimension of the model and is the size of the initial token embeddings
        num_heads: This is the number of attention blocks that are used
        hidden_dim: This is the dimension of the hidden layer in the MLP
        num_layers: The number of decoder blocks to stack
        mask: If not None, then this is a matrix of boolean values that indicate which elements in the attention scores to mask
        dropout: This is the percentage of model weights to randomly not update
        max_seq_len: The largest sequence context window that the model can process
    """
    def __init__(self, vocab_size, model_dim=256, num_heads=8, hidden_dim=512, num_layers=4, dropout=0.1, max_seq_len=512):
        super().__init__()
        #The nn.Embedding takes the vocabulary size and the desired dimension and converts each token into an initial embedding via a lookup table
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_seq_len)
        self.layers = nn.ModuleList([TransformerDecoderLayer(model_dim, num_heads, hidden_dim, dropout) for block in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.lay_norm = nn.LayerNorm(model_dim)
        self.out_linear = nn.Linear(model_dim, vocab_size)
      
    def forward(self, x, mask):
        x = self.token_embedding(x) #Initial token embedding
        x = self.positional_encoding(x) #Incorporate positional information
        out = self.dropout(x)
        #Feed the input through the decoder layers
        for layer in self.layers:
            out = layer(out, mask)
        out = self.lay_norm(out)
        #Utilize a final MLP to project the embeddings to the vocabulary size to determine probabilities for each token position
        logits = self.out_linear(out) # Output shape: (batch_size, seq_len, vocab_size)
        return logits
        



##############################################################################################################################################################
##################################################################### Custom Text Dataset ####################################################################
##############################################################################################################################################################
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """
    This module forms a dataset from some data and prepares examples that can be sampled from raw tokenized data.
    Each sample x will be a sequence of length "seq_len" and its corresponding target sequence y will simply
    be x with positions shifted by 1 forward (this is because we are trying to do next-token prediction)
    """
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        #This returns the total number of training examples that are possible
        return (len(self.data) - 1) // self.seq_len
    
    def __getitem__(self, index):
        #This function returns a new sampled sequence of text data
        i = index * self.seq_len #This is the "beginning of the sequence" index
        #Sample a new sequence from the text data
        x = self.data[i : i + self.seq_len]
        #Determine the corresponding target sequence (which is simply the x sequence shifted by 1 token)
        y = self.data[i+1 : i + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y) #Returns a sample (x,y) sequence pair

def build_vocab(text):
    """
    Build a vocabulary mapping (token-to-index and index-to-token) from the provided text.
    For simplicity, we split on whitespace. Note the following definitions.
        - vocab: Maps tokens (words) to their unique numerical indices
        - inv_vocab: The inverted vocabulary which maps indices back to their corresponding tokens
    """
    cleaned_text = re.sub(r'[\d\[\]\n\t*]', '', text)
    char_data = list(cleaned_text)
    tokens = sorted(list(set(cleaned_text))) #Creates a list of characters
    #tokens = cleaned_text.split() # splits the text based on whitespace. This will be a list of words
    vocab = {} #Vocabulary hash map (dictionary)
    #Map every token to a unique dictionary value
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    inv_vocab = {idx: token for token, idx in vocab.items()}
    # Convert the full text into a list of token indices
    data = [vocab[char] for char in char_data]
    #data = [vocab[token] for token in tokens]
    return vocab, inv_vocab, data

def Set_Seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)





##############################################################################################################################################################
###################################################################### Training Algorithm ####################################################################
##############################################################################################################################################################
def train_algo(model, dataloaders, optimizer, loss_func, device, num_epochs, lr_scheduler, sched_type):
    """
    Training algorithm
    """
    #Set up an interactive plot
    plt.ion()
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    line, = ax.plot(xdata, ydata, '-')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training Transformer Decoder Language Model', size=15)
    ax.grid(True)
    
    train_dataloader = dataloaders[0]
    val_dataloader = dataloaders[1]
    iter_list = []
    loss_list = []
    val_loss_list = []
    i = 0
    for epoch in range(num_epochs):
        epoch_error_list = []
        for (x,y) in train_dataloader:
            model.train()
            iter_list.append(i)
            i+=1
            
            x, y = x.to(device), y.to(device)
            #Create a look-ahead mask
            seq_len = x.size(1)
            mask = generate_look_ahead_mask(seq_len).to(device)
            #Model prediction
            preds = model(x, mask)
            # Reshape preds and targets for computing cross-entropy loss:
            # preds: (batch_size * seq_len, vocab_size)
            # targets: (batch_size * seq_len)
            loss = loss_func(preds.view(-1, preds.size(-1)), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            epoch_error_list.append(loss.item())
            
            if sched_type == "one_cycle":
                lr_scheduler.step()
            
            #Compute the validation loss
            model.eval()
            with torch.no_grad():
                x_val, y_val = next(iter(val_dataloader))
                x_val, y_val = x_val.to(device), y_val.to(device)
                seq_len = x_val.size(1)
                mask = generate_look_ahead_mask(seq_len).to(device)
                #Model prediction
                val_preds = model(x_val, mask)
                val_loss = loss_func(val_preds.view(-1, val_preds.size(-1)), y_val.view(-1))
                val_loss_list.append(val_loss.item())
            
            #Live plotting
            xdata.append(i)
            ydata.append(loss.item())
            line.set_xdata(xdata)
            line.set_ydata(ydata)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        avg_loss = np.sum(np.asarray(epoch_error_list)) / len(epoch_error_list)
        if sched_type == "plateau":
            lr_scheduler.step(avg_loss)
        #print(optimizer.param_groups[0]['lr'])
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")
        plt.ioff() #Turn the interactive plot off
    return iter_list, loss_list, val_loss_list
            
        


##############################################################################################################################################################
####################################################################### Text Generation ######################################################################
##############################################################################################################################################################
def generate_text_word(model, start_text, vocab, inv_vocab, max_length, device):
    """
    Given a starting string, generate text using the trained model.
    At each step, the model predicts the next token, which is appended to the sequence.
    This is a relatively standard text-generation function
    """
    model.eval()
    # Tokenize the starting text
    tokens = [vocab[token] for token in start_text.split() if token in vocab]
    input_seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # Shape: (1, seq_len)
    for _ in range(max_length):
        seq_len = input_seq.size(1)
        mask = generate_look_ahead_mask(seq_len).to(device)
        logits = model(input_seq, mask)
        # Focus on the logits for the last token in the sequence
        next_token_logits = logits[0, -1, :]
        # Greedily choose the token with the highest probability (alternatively, sample for diversity)
        next_token = torch.argmax(next_token_logits).item()
        # Append the predicted token to the sequence
        input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=device)], dim=1)
    # Convert token indices back to words
    generated_tokens = input_seq[0].tolist()
    generated_text = " ".join([inv_vocab[t] for t in generated_tokens])
    return generated_text

def generate_text_char(model, start_text, vocab, inv_vocab, max_length, device):
    """
    Generate text one character at a time using a character-level model.
    This is a relatively standard text-generation function
    """
    model.eval()
    # Convert the starting text to a list of token indices (characters)
    tokens = [vocab[char] for char in start_text if char in vocab]
    # Create a tensor of shape (1, seq_len) for the model input
    input_seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate one character at a time until reaching max_length
    for _ in range(max_length):
        seq_len = input_seq.size(1)
        # Create a causal mask to ensure that the model cannot "see" future characters
        mask = generate_look_ahead_mask(seq_len).to(device)
        # Get model predictions (logits) for the current sequence
        logits = model(input_seq, mask)
        # Focus on the logits for the last character in the sequence
        next_token_logits = logits[0, -1, :]
        # Use greedy decoding: choose the character with the highest probability
        next_token = torch.argmax(next_token_logits).item()
        # Append the predicted token to the input sequence for the next iteration
        input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=device)], dim=1)
    
    # Convert the token indices back to characters and join them to form the output string
    generated_chars = [inv_vocab[token] for token in input_seq[0].tolist()]
    generated_text = "".join(generated_chars)
    return generated_text




##############################################################################################################################################################
############################################################################ Main ############################################################################
##############################################################################################################################################################
def main():
    ###########################################################################
    ########################### Define Parameters #############################
    ###########################################################################
    seq_len = 256 #The number of consecutive tokens in a single example sequence (this is essentially the length of a sentence)
    batch_size = 64 #The number of sequences to proccess in parallel
    model_dim = 384 #The dimension of the initial embedding (the dimension of the input vector passed to the model)
    num_heads = 6 #The number of attention heads in a multi-headed attention block
    hidden_dim = 1500 #The size of the hidden layer in the MLP layers
    num_layers=6 #The number of decoder blocks to stack
    dropout=0.2 #The percentage of random weights that are not updated
    max_seq_len = seq_len
    learning_rate = 3e-4 #3e-4
    num_epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Set_Seed(42)
    
    ###########################################################################
    ####################### Data Loading & Preprocessing ######################
    ###########################################################################
    #Load in the text data
    with open('INSERT PATH HERE', 'r', encoding='utf-8') as f:
        text = f.read()
    #Build the vocabulary in the text file
    print("Building vocabulary...")
    vocab, inv_vocab, data = build_vocab(text)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    #Split the data into a training set and a validation set
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    
    #Create the training dataset and dataloader
    train_dataset = TextDataset(train_data, seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #Create the validation dataset and dataloader
    val_dataset = TextDataset(val_data, seq_len)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True)
    dataloaders = [train_dataloader, val_dataloader] #Put them both into a list
    
    
    ###########################################################################
    ######################### Training and Generation #########################
    ###########################################################################
    #Define the Model and the Optimizer
    model = TransformerLanguageModel(vocab_size, model_dim, num_heads, hidden_dim, num_layers, dropout, max_seq_len).to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, epochs=num_epochs, steps_per_epoch=len(train_dataloader)) #1e-2
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=False)
    sched_type = "one_cycle" #"none" #"one_cycle" #"plateau"
    
    #Train the model
    iter_list, train_loss_list, val_loss_list = train_algo(model, dataloaders, optimizer, loss_func, device, num_epochs, lr_scheduler, sched_type)
    #Plotting
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    sns.lineplot(x=iter_list, y=train_loss_list, linewidth=0.3, label='Training Loss', legend=True, color='blue', linestyle='-').set_title('Austen-Net: Trained Transformer Decoder Language Model', size=15)
    sns.lineplot(x=iter_list, y=val_loss_list, linewidth=0.3, label='Validation Loss', legend=True, color='red', linestyle='-')
    plt.xlabel("Iterations")
    plt.ylabel("Cross-Entropy Loss")
    axes.grid(True)
    plt.show()
    
    #Generate text from the trained model
    seed_text = "With the sunrise came a feeling of longing"  # A sample seed text
    size_generated_text = max_seq_len - len(seed_text)
    #generated = generate_text_word(model, seed_text, vocab, inv_vocab, max_length=size_generated_text, device=device)
    generated = generate_text_char(model, seed_text, vocab, inv_vocab, max_length=size_generated_text, device=device)
    print("Seed text:\n", seed_text)
    print("\nGenerated Text:\n", generated)





if __name__ == "__main__":
    main()




