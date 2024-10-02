import pickle

import torch
import torch.nn as nn
import numpy as np
import gym
import gym_compete
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, random_split, DataLoader

ob_mean = np.load("parameters/human-to-go/obrs_mean.npy")
ob_std = np.load("parameters/human-to-go/obrs_std.npy")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_seq_length=10, num_layers=2):
        super(LSTMPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.max_seq_length = max_seq_length
        self.sequence_buffer = []

    def forward(self, x, lengths=None, testing_mode=False, reset_to_initial_state=False):
        if reset_to_initial_state is True:
            self._reset_to_initial_state()

        if testing_mode is True:
            assert x.shape[0] == 1, "Batch_size==1"
            assert x.shape[1] == 1, "Only one time step at a time"
            self.sequence_buffer.append(x)
            if len(self.sequence_buffer) > self.max_seq_length:
                self.sequence_buffer.pop(0)

            # original version
            # Stack the buffered sequences and ensure correct shape
            # current_sequence = torch.cat(self.sequence_buffer, dim=1)
            # lengths = torch.tensor([len(self.sequence_buffer)])
            #
            # # Pack and pad the sequence
            # packed_x = nn.utils.rnn.pack_padded_sequence(current_sequence, lengths, batch_first=True,
            #                                              enforce_sorted=False)
            #
            # packed_out, _ = self.lstm(packed_x.to(device))
            # lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            # out = self.fc(lstm_out)

            # simplified
            current_sequence = torch.cat(self.sequence_buffer, dim=1).to(device)
            lstm_out, _ = self.lstm(current_sequence)
            out = self.fc(lstm_out)
            return out[:, -1, :]  # Only take the output of the last time step
        else:
            # Extract original lengths before padding
            if lengths is None:
                lengths = torch.tensor([seq.size(0) for seq in x])

                # Padding sequences to max_seq_length
                padded_sequences = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                                             padding_value=0.0)
            # if padded_sequences.size(1) < self.max_seq_length:
            #     # Add additional padding to reach max_seq_length
            #     padding = torch.zeros((padded_sequences.size(0), self.max_seq_length - padded_sequences.size(1),
            #                            padded_sequences.size(2)))
            #     padded_sequences = torch.cat([padded_sequences, padding], dim=1)
            # elif padded_sequences.size(1) > self.max_seq_length:
            #     # Trim the sequence if it exceeds max_seq_length
            #     padded_sequences = padded_sequences[:, :self.max_seq_length, :]
            else:
                padded_sequences = x
            packed_x = nn.utils.rnn.pack_padded_sequence(padded_sequences, lengths, batch_first=True,
                                                         enforce_sorted=False)

            packed_out, _ = self.lstm(packed_x.to(device))
            lstm_out, info = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

            # Only take the valid part of lstm_out for each sequence
            valid_lstm_out = []
            for i, length in enumerate(lengths):
                valid_lstm_out.append(lstm_out[i, :length])

            valid_lstm_out = torch.cat(valid_lstm_out, dim=0)

            # Apply fully connected layer to valid LSTM outputs
            out = self.fc(valid_lstm_out)
            # print("out", out.shape)
            # Return the last valid output for each sequence
            last_valid_out = out.new_zeros(len(lengths), out.size(1))
            # print(last_valid_out.shape)
            # print(lengths)
            for i, length in enumerate(lengths):
                last_valid_out[i] = out[lengths[:i + 1].sum().item() - 1]
                # print(lengths[:i + 1].sum().item() - 1)

            return last_valid_out

    def predict(self, x, reset=False):
        if reset:
            self._reset_to_initial_state()
        self.eval()
        # assert x.shape[0] == 1, "Batch_size==1"
        # assert x.shape[1] == 1, "Only one time step at a time"
        # self.sequence_buffer.append(x)
        # if len(self.sequence_buffer) > self.max_seq_length:
        #     self.sequence_buffer.pop(0)
        # current_sequence = torch.cat(self.sequence_buffer, dim=1).to(device)
        # lstm_out, _ = self.lstm(current_sequence)
        # out = self.fc(lstm_out)
        # return out[:, -1, :].detach()
        out = self.forward(x, testing_mode=False).detach()
        # print("out", out.shape)
        return out

    def _reset_to_initial_state(self):
        self.sequence_buffer = []
