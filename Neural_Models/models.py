import paddle 
import paddle.nn as nn
import paddle.nn.functional as F


class BoW(nn.Layer):

    def __init__(self, 
                vocab_size, 
                output_dim,
                embedding_dim=100,
                padding_idx=0,  
                hidden_dim=50, 
                activation=nn.ReLU()):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.dense = nn.Linear(embedding_dim, hidden_dim)
        self.activation = activation
        self.dense_out = nn.Linear(hidden_dim, output_dim)

    def encoder(self, embd):
        return embd.sum(axis=1)

    def forward(self, text_ids): 
        text_embd = self.embedding(text_ids)
        encoded = self.encoder(text_embd)
        hidden_out = self.activation(self.dense(encoded))
        out_logits = self.dense_out(hidden_out)
        return out_logits


class CNN(nn.Layer):

    def __init__(self,
                 vocab_size,
                 output_dim,
                 embedding_dim=100,
                 padding_idx=0,
                 num_filter=256,
                 filter_sizes=(1,),
                 hidden_dim=50,
                 activation=nn.ReLU()):
        
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.convs = nn.LayerList([
            nn.Conv1D(
                in_channels=embedding_dim,
                out_channels=num_filter,
                kernel_size=fz
            ) for fz in filter_sizes
        ])
        self.dense = nn.Linear(len(filter_sizes) * num_filter, hidden_dim)
        self.activation = activation
        self.dense_out = nn.Linear(hidden_dim, output_dim)
    
    def encoder(self, embd):
        embd = embd.transpose((0,2,1))
        conved = [self.activation(conv(embd)) for conv in self.convs]
        max_pooled = [F.adaptive_max_pool1d(conv, output_size=1).squeeze(2) for conv in conved]
        pooled_concat = paddle.concat(max_pooled, axis=1)
        return pooled_concat
 
    def forward(self, text_ids):
        text_embd = self.embedding(text_ids)
        encoded = self.encoder(text_embd)
        hidden_out = self.activation(self.dense(encoded))
        out_logits = self.dense_out(hidden_out)
        return out_logits


class LSTM(nn.Layer):

    def __init__(self,
                 vocab_size,
                 output_dim,
                 embedding_dim=100,
                 lstm_hidden_dim=128,
                 padding_idx=0,
                 hidden_dim_out=50,
                 n_layers=1,
                 bidirectional=False,
                 dropout_rate=0.0,
                 activation=nn.ReLU()):
        
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.direction = 'bidirect' if bidirectional is True else 'forward'
        self.lstm = nn.LSTM(
            embedding_dim, lstm_hidden_dim, n_layers, self.direction, dropout=dropout_rate)
        lstm_out_dim = lstm_hidden_dim * 2 if bidirectional is True else lstm_hidden_dim
        self.dense = nn.Linear(lstm_out_dim, hidden_dim_out)
        self.activation = activation
        self.dense_out = nn.Linear(hidden_dim_out, output_dim)


    def encoder(self, embd, seq_len):
        encoded, (hidden, cell) = self.lstm(embd, sequence_length=seq_len)
        if self.direction != 'bidirect':
            return hidden[-1, :, :]
        return paddle.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)

    def forward(self, text_ids, seq_len):
        text_embd = self.embedding(text_ids)
        encoded = self.encoder(text_embd, seq_len)
        hidden_out = self.activation(self.dense(encoded))
        out_logits = self.dense_out(hidden_out)
        return out_logits


class LogisticRegression(nn.Layer):

    def __init__(self, 
                vocab_size, 
                output_dim,
                embedding_dim=100,
                padding_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.dense = nn.Linear(embedding_dim, output_dim)

    def encoder(self, embd):
        return embd.sum(axis=1)

    def forward(self, text_ids): 
        text_embd = self.embedding(text_ids)
        encoded = self.encoder(text_embd)
        out_logits = self.dense(encoded)
        return out_logits

