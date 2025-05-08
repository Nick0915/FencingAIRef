
# ! #THIS FILE SHOULD NOT BE RUN!!! IT IS A DEPENDENCY FOR ANOTHER FILE

import torch
from torch.nn.modules import RNN, RNNCell, LSTM, LSTMCell, GRU, GRUCell, Linear, Dropout

class MultiLSTM(torch.nn.Module):
    def __init__(self, input_size, num_lstm_layers, lstm_hidden_size, dropout=0.):
        super(MultiLSTM, self).__init__()

        self.input_size = input_size
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout = dropout

        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = Dropout(dropout)

        self.linear = Linear(lstm_hidden_size, 1) # take feature vec and turn it into 1 activation

    def forward(self, X):
        # batch_size, seq_len, _ = X.shape
        batch_size = len(X.sorted_indices)

        # lstm_hidden_state_0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size)
        # lstm_context_state_0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size)
        lstm_output, (lstm_hidden_state_f, lstm_context_state_f) = self.lstm(
            X#, (lstm_hidden_state_0, lstm_context_state_0)
        )

        lstm_output, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)

        return self.linear(lstm_output[:, -1, :])



class LSTM_MultiRNN(torch.nn.Module):
    def __init__(self, input_size, num_rnn_layers, lstm_hidden_size, rnn_hidden_size, dropout=0.):
        super(LSTM_MultiRNN, self).__init__()

        self.input_size = input_size
        self.num_rnn_layers = num_rnn_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout

        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            # dropout=dropout
        )

        self.mid_dropout = Dropout(dropout)

        self.rnn = RNN(
            input_size=lstm_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.linear = Linear(rnn_hidden_size, 1) # take feature vec and turn it into activations for 2 possible states

    def forward(self, X):
        # batch_size, seq_len, _ = X.shape
        batch_size = len(X.sorted_indices)

        # lstm_hidden_state_0 = torch.zeros(1, batch_size, self.lstm_hidden_size)
        # lstm_context_state_0 = torch.zeros(1, batch_size, self.lstm_hidden_size)
        lstm_output, (lstm_hidden_state_f, lstm_context_state_f) = self.lstm(
            X#, (lstm_hidden_state_0, lstm_context_state_0)
        )

        # rnn_hidden_state_0 = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size)
        rnn_output, rnn_hidden_state_f = self.rnn(
            lstm_output
            #, rnn_hidden_state_0
        )

        rnn_output, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

        return self.linear(rnn_output[:, -1, :])

