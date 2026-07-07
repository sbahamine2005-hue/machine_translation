import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer


#tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = tokenizer.vocab_size
PAD_ID = tokenizer.pad_token_id 
BOS_ID = tokenizer.bos_token_id

#embedding_dim=256, hidden_stat_dim=512, input_dim=VOCAB_SIZE
class Encoder(nn.Module):
    def __init__(self, embedding_dim: int=256, hidden_stat_dim: int=512, input_dim: int=vocab_size):
        super().__init__()
        # input_dim = total size of vocabulary
        # embading_dim = size of the dense word vectors
        self.embading = nn.Embedding(input_dim,embedding_dim= embedding_dim, padding_idx=tokenizer.pad_token_id)
        # The LSTM's input_size must match the embedding_dim
        self.lstm = nn.LSTM(input_size= embedding_dim, hidden_size= hidden_stat_dim,batch_first=True)

    def forward(self, input):
        # input shape: (batch_size, sequence_length)
        embading_input = self.embading(input)
        # embading_input shape: (batch_size, sequence_length, embading_dim)
        
        # LSTM processes the entire sequence
        output , hidden = self.lstm(embading_input)

        return hidden 
    
#Now the decoder parte 
class Decoder(nn.Module):
    def __init__(self,output_dim: int=vocab_size, embedding_dim: int=256, hidden_stat_dim: int=512):#output_dim, emb_dim, hidden_dim
        super().__init__()
        self.embading = nn.Embedding(output_dim, embedding_dim= embedding_dim, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_stat_dim, batch_first=True)
        self.linear = nn.Linear(hidden_stat_dim, output_dim)
       

    def forward(self, input_token, previous_hidden):
        # input_token shape: (batch_size) -> unsqueeze to (batch_size, 1) for seq_len=1
        input_token = input_token.unsqueeze(1)
        # embedded shape: (batch_size, 1, embading_dim)
        embading = self.embading(input_token)
        output, hidden_state = self.lstm(embading, previous_hidden)
       
        # output shape: (batch_size, 1, hidden_stat_dim)
        # Squeeze sequence dimension out before sending to Linear layer
        prediction = self.linear(output.squeeze(1))
        # prediction shape: (batch_size, output_dim) -> Raw continuous logits
        
        return prediction, hidden_state  




class Seq2Seqmodel(nn.Module):
    def __init__(self, encoder,decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, targ = None, teacher_forcing_ratio=0.5):
        # source shape: (batch_size, source_seq_len)
        # target shape: (batch_size, target_seq_len)
        batch_size = source.shape[0]

        max_length = targ.shape[1] if targ is not None else 20
        vocab_size = self.decoder.linear.out_features

        # We must collect the continuous logits tensor to calculate loss later
        outputs = torch.zeros(max_length, batch_size, vocab_size).to(self.device) 

        # Get initial context tuple (h_n, c_n) from encoder
        hidden= self.encoder(source)#context_vector = hidden
        # FIXED: Use the exact dynamic BOS token ID filled with the correct type
        input_token = torch.full((batch_size,), BOS_ID, dtype=torch.long).to(self.device)

        for i in range(max_length):
            prediction, hidden_state  = self.decoder(input_token, hidden)
            # Store continuous predictions: shape (batch_size, vocab_size) at time t
            outputs[i] = prediction

            #the next hidden state is the predicted one
            hidden = hidden_state
            
            top_token = prediction.argmax(1)
            

            if targ is not None and torch.rand(1).item()<teacher_forcing_ratio : #so like that i have a 50% force teaching
                input_token = targ[:,i]# Feed actual ground truth token
            else: 
                input_token = top_token # Feed model's own guess

        #the output format is (max_length, batch_size, vocab_size)    
        
        # Permute outputs to return a standard shape: (batch_size, max_length, vocab_size)
        return outputs.permute(1, 0, 2)

    

