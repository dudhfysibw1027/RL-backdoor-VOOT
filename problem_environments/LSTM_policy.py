import os
import pickle

import torch
import torch.nn as nn
import numpy as np
import gym
import gym_compete
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, random_split, DataLoader

now_path = os.getcwd()
print(os.getcwd())
if 'problem_environments' not in now_path:
    ob_mean = np.load("../problem_environments/parameters/human-to-go/obrs_mean.npy")
    ob_std = np.load("../problem_environments/parameters/human-to-go/obrs_std.npy")
else:
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


class LSTMPolicyMultiDiscrete(nn.Module):
    def __init__(self, input_size, hidden_size, action_space, max_seq_length=5, num_layers=2, seed=42):
        super(LSTMPolicyMultiDiscrete, self).__init__()
        torch.manual_seed(seed)

        # Network
        self.head = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, sum(action_space.nvec))

        # Store hyperparameters
        self.max_seq_length = max_seq_length
        self.action_space = action_space
        self.hidden_size = hidden_size

        # Cosine embedding
        self.K = 32
        self.n_tau = 8
        self.n_cos = 64
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(
            device)  # Starting from 0 as in the paper
        self.cos_embedding = nn.Linear(self.n_cos, hidden_size)

        # For inference
        self.sequence_buffer = []

    def calc_cos(self, batch_size, seq_len, n_tau):
        """
        Generate cosine embeddings for tau values across sequence length.
        """
        taus = torch.rand(batch_size, seq_len, n_tau, 1).to(self.pis.device)  # (batch, seq_len, n_tau, 1)
        cos = torch.cos(taus * self.pis)  # (batch, seq_len, n_tau, n_cos)
        return cos, taus

    def forward(self, x, lengths=None, testing_mode=True, reset_to_initial_state=False, num_tau=8):
        """
        Two modes, train when testing_mode=False, inference when testing_mode=True
        """
        batch_size = x.shape[0]

        if reset_to_initial_state is True:
            self._reset_to_initial_state()

        if testing_mode is True:
            # assert x.shape[0] == 1, "Batch_size==1"
            # assert x.shape[1] == 1, "Only one time step at a time"

            # for i, t in enumerate(self.sequence_buffer):
            #     print(f"Buffer tensor {i} is on {t.device}")
            # print(current_sequence.shape)
            seq_len = x.shape[1]
            x = torch.relu(self.head(x))

            # Generate cosine embeddings
            cos, taus = self.calc_cos(batch_size, seq_len,
                                      num_tau)  # (batch, seq_len, n_tau, n_cos)
            cos_x = torch.relu(self.cos_embedding(cos))  # (batch, seq_len, n_tau, layer_size)

            # Expand state embedding for element-wise multiplication
            x = x.unsqueeze(2)  # (batch, seq_len, 1, layer_size)
            x = (x * cos_x).view(batch_size * num_tau, seq_len,
                                 self.hidden_size)  # (batch, seq_len, n_tau * layer_size)

            lstm_out, _ = self.lstm(x)
            lstm_out_tau = lstm_out.view(batch_size, num_tau, seq_len, -1)
            # take mean for every tau value
            lstm_out = lstm_out_tau.mean(dim=1)

            out = self.hidden_layer(lstm_out)
            out = torch.nn.functional.relu(out)
            out = self.output_layer(out)
            logits = out[:, -1, :]  # Only take the output of the last time step

            # print("logits/action_logits")
            # print(logits[0:3])
            action_logits = torch.split(logits, self.action_space.nvec.tolist(), dim=-1)  # 切成多個 (batch_size, n) 張量
            # print(action_logits)
            actions = [torch.argmax(logit, dim=-1) for logit in action_logits]
            # distribution = [Categorical(logits=split) for split in action_logits]

            action_argmax = torch.stack(actions, dim=-1)
            # action_sample = torch.stack([dist.sample() for dist in distribution], dim=1)
            # print("distribution sample/argmax")
            # print(action_sample)
            # print(action_argmax)
            return action_argmax
        else:
            seq_len = x.shape[1]
            # print("before", x.shape)  # before torch.Size([1, 1, 18])
            x = torch.relu(self.head(x))
            # print("after", x.shape)  # after torch.Size([1, 1, 128])
            # Generate cosine embeddings
            cos, taus = self.calc_cos(batch_size, seq_len, self.n_tau)  # (batch, seq_len, n_tau, n_cos)
            cos_x = torch.relu(self.cos_embedding(cos))  # (batch, seq_len, n_tau, layer_size)
            # print("cos", cos.shape, "cos_x", cos_x.shape)  # cos torch.Size([1, 1, 8, 64]) cos_x torch.Size([1, 1, 8, 128])
            # Expand state embedding for element-wise multiplication
            x = x.unsqueeze(2)  # (batch, seq_len, 1, layer_size)
            # print(x.shape)
            x = (x * cos_x).view(batch_size * self.n_tau, seq_len,
                                 self.hidden_size)  # (batch, seq_len, n_tau * layer_size)
            # print(x.shape)
            # Extract original lengths before padding
            if lengths is None:
                lengths = torch.tensor([seq_len])
                # Padding sequences to max_seq_length
                padded_sequences = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                                             padding_value=0.0)
            else:
                padded_sequences = x
            # print(lengths.shape)
            # print(f"padded_sequences.shape: {padded_sequences.shape}")  # 應該是 (batch_size, max_seq_length, feature_dim)
            expanded_lengths = lengths.repeat_interleave(self.n_tau)
            # print(f"expanded_lengths: {expanded_lengths}")  # 每個序列的有效長度
            # print(f"max(expanded_lengths): {expanded_lengths.max()}")  # 是否超過 padded_sequences.shape[1]

            # print(expanded_lengths[-40:-30])
            # packed_x = nn.utils.rnn.pack_padded_sequence(padded_sequences, expanded_lengths, batch_first=True,
            #                                              enforce_sorted=False)

            # packed_out, _ = self.lstm(packed_x)
            lstm_out, _ = self.lstm(x)

            # lstm_out, info = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            # print(lstm_out.shape)
            lstm_out_tau = lstm_out.view(batch_size, self.n_tau, seq_len, -1)

            # print("lstm_out_tau", lstm_out_tau.shape)
            lstm_out = lstm_out_tau.mean(dim=1)
            # print("lstm_out", lstm_out.shape)
            # print(lstm_out_tau.shape, lstm_out.shape)
            # Only take the valid part of lstm_out for each sequence
            valid_lstm_out = []
            for i, length in enumerate(lengths):
                print(i, length)
                valid_lstm_out.append(lstm_out[i, :length])

            valid_lstm_out = torch.cat(valid_lstm_out, dim=0)

            # Apply fully connected layer to valid LSTM outputs
            out = self.hidden_layer(valid_lstm_out)
            out = torch.nn.functional.relu(out)
            out = self.output_layer(out)
            # print("out", out.shape)
            # print(out[1][3:6])
            # Split logits for each action dimension
            action_logits = torch.split(out, self.action_space.nvec.tolist(),
                                        dim=-1)
            # print(len(action_logits))  # = 5
            # print(action_logits[0].shape)  # = torch.Size([619, 4])
            # print(action_logits[1][1])  # [0][1] equals action_dim=0(logits[0:3]) all_batch_idx=1
            # [1][1] equals action_dim=1(logits[3:6]) all_batch_idx=1, out[1][3:6]
            # Convert logits to probabilities using softmax
            # probabilities = [torch.softmax(logit, dim=-1) for logit in action_logits]
            probabilities = action_logits
            # Extract the last valid output for each sequence
            last_valid_probabilities = []
            # print(lengths)
            # tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            #         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,  9,  9,  9,
            #          9,  8,  8,  8,  8,  8,  8,  8,  7,  6,  6,  6,  6,  6,  5,  5,  5,  4,
            #          4,  3,  3,  3,  3,  2,  2,  2,  1,  1])
            # action space: MultiDiscrete([4 4 4 4 4])
            for i, length in enumerate(lengths):
                # For each dimension, extract the last valid probability
                # torch.Size([5, 4])
                last_probs = [prob[lengths[:i + 1].sum().item() - 1] for prob in probabilities]
                last_valid_probabilities.append(
                    torch.stack(last_probs, dim=0))  # Stack to get shape (action_dims, num_classes)

            # last_valid_probabilities
            # torch.Size([64, 5, 4])

            # Stack all sequences into a tensor: shape (batch_size, action_dims, num_classes)
            last_valid_probabilities = torch.stack(last_valid_probabilities, dim=0)
            # print(last_valid_probabilities.shape)

            # Return the probabilities of the last valid actions
            return last_valid_probabilities

    def predict(self, x, reset=False):
        if reset:
            self._reset_to_initial_state()
        self.eval()
        with torch.no_grad():
            out = self.forward(x, testing_mode=True, num_tau=self.K)
            out = out.cpu().numpy().squeeze()
        # out = self.forward(x, testing_mode=True).detach()

        return out

    def _reset_to_initial_state(self):
        self.sequence_buffer = []


class LSTMPolicyMultiDiscrete_(nn.Module):
    def __init__(self, input_size, hidden_size, action_space, max_seq_length=5, num_layers=2, seed=42):
        super(LSTMPolicyMultiDiscrete_, self).__init__()
        torch.manual_seed(seed)

        # Network
        self.head = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, sum(action_space.nvec))
        self.attn_layer = nn.Linear(hidden_size, hidden_size)

        # Store hyperparameters
        self.max_seq_length = max_seq_length
        self.action_space = action_space
        self.hidden_size = hidden_size

        # Cosine embedding
        self.K = 32
        self.n_tau = 8
        self.n_cos = 64
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(
            device)  # Starting from 0 as in the paper
        self.cos_embedding = nn.Linear(self.n_cos, hidden_size)

        # For inference
        self.sequence_buffer = []

    def calc_cos(self, batch_size, seq_len, n_tau):
        """
        Generate cosine embeddings for tau values across sequence length.
        """
        taus = torch.rand(batch_size, seq_len, n_tau, 1).to(self.pis.device)  # (batch, seq_len, n_tau, 1)
        cos = torch.cos(taus * self.pis)  # (batch, seq_len, n_tau, n_cos)
        return cos, taus

    def forward(self, x, lengths=None, testing_mode=True, reset_to_initial_state=False, num_tau=8):
        """
        Two modes, train when testing_mode=False, inference when testing_mode=True
        """
        batch_size = x.shape[0]

        if reset_to_initial_state is True:
            self._reset_to_initial_state()

        if testing_mode is True:
            seq_len = x.shape[1]
            # print("model_input_shape", x.shape)
            x = torch.relu(self.head(x))

            # Generate cosine embeddings
            cos, taus = self.calc_cos(batch_size, seq_len,
                                      num_tau)  # (batch, seq_len, n_tau, n_cos)
            cos_x = torch.relu(self.cos_embedding(cos))  # (batch, seq_len, n_tau, layer_size)

            # Expand state embedding for element-wise multiplication
            x = x.unsqueeze(2)  # (batch, seq_len, 1, layer_size)
            x = (x * cos_x).view(batch_size * num_tau, seq_len,
                                 self.hidden_size)  # (batch, seq_len, n_tau * layer_size)

            lstm_out, (h_n, c_n) = self.lstm(x)
            lstm_out_tau = lstm_out.view(batch_size, num_tau, seq_len, -1)
            # take mean for every tau value
            lstm_out = lstm_out_tau.mean(dim=1)
            query = h_n[-1].view(batch_size, num_tau, -1)
            query = query.mean(dim=1)
            # print("h_n", query.shape)  # h_n torch.Size([1, 512])
            query = self.attn_layer(query)

            query = query.unsqueeze(1)  # [1, 1, hidden_size]
            scores = (lstm_out * query).sum(dim=2)

            attn_weights = torch.nn.functional.softmax(scores, dim=1)  # [1, seq_len]
            attn_weights = attn_weights.unsqueeze(2)  # => [1, seq_len, 1]

            context = torch.sum(lstm_out * attn_weights, dim=1)

            # (optional) hidden_layer
            context = self.hidden_layer(context)
            # out_layer
            out = self.output_layer(context)  # [1, sum(action_space.nvec)]
            # e.g. MultiDiscrete([3,3,3]) => total=9
            action_logits = torch.split(out, self.action_space.nvec.tolist(), dim=-1)
            # => list of Tensors, each shape=[1,3]
            # 取argmax => shape [1] => stack => shape [1,3]
            actions = [torch.argmax(logit, dim=-1) for logit in action_logits]
            action_argmax = torch.stack(actions, dim=-1)
            # return shape => [1, 3]
            return action_argmax

            # out = self.hidden_layer(lstm_out)
            # out = torch.nn.functional.relu(out)
            # out = self.output_layer(out)
            # logits = out[:, -1, :]  # Only take the output of the last time step

            # print("logits/action_logits")
            # print(logits[0:3])
            # action_logits = torch.split(logits, self.action_space.nvec.tolist(), dim=-1)  # 切成多個 (batch_size, n) 張量
            # print(action_logits)
            # actions = [torch.argmax(logit, dim=-1) for logit in action_logits]
            # distribution = [Categorical(logits=split) for split in action_logits]

            # action_argmax = torch.stack(actions, dim=-1)
            # action_sample = torch.stack([dist.sample() for dist in distribution], dim=1)
            # print("distribution sample/argmax")
            # print(action_sample)
            # print(action_argmax)
            # return action_argmax
        else:
            # print("before", x.shape)  # before torch.Size([512, 5, 18])
            x = torch.relu(self.head(x))
            # print("after", x.shape)  # after torch.Size([512, 5, 128])
            # Generate cosine embeddings
            cos, taus = self.calc_cos(batch_size, self.max_seq_length, self.n_tau)  # (batch, seq_len, n_tau, n_cos)
            cos_x = torch.relu(self.cos_embedding(cos))  # (batch, seq_len, n_tau, layer_size)

            # Expand state embedding for element-wise multiplication
            x = x.unsqueeze(2)  # (batch, seq_len, 1, layer_size)
            x = (x * cos_x).view(batch_size * self.n_tau, self.max_seq_length,
                                 self.hidden_size)  # (batch, seq_len, n_tau * layer_size)
            # Extract original lengths before padding
            if lengths is None:
                lengths = torch.tensor([seq.size(0) for seq in x])
                # Padding sequences to max_seq_length
                padded_sequences = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                                             padding_value=0.0)
            else:
                padded_sequences = x
            # print(lengths.shape)
            expanded_lengths = lengths.repeat_interleave(self.n_tau)
            # print(expanded_lengths[-40:-30])
            packed_x = nn.utils.rnn.pack_padded_sequence(padded_sequences, expanded_lengths, batch_first=True,
                                                         enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed_x)

            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            # print(lstm_out.shape)
            lstm_out_tau = lstm_out.view(batch_size, self.n_tau, self.max_seq_length, -1)

            lstm_out = lstm_out_tau.mean(dim=1)
            # print("tau", lstm_out_tau.shape, lstm_out.shape)

            query = h_n[-1].view(batch_size, self.n_tau, -1)
            query = query.mean(dim=1)
            # print("h_n", query.shape)  # h_n torch.Size([512, 128])
            query = self.attn_layer(query)
            # Establish mask
            mask = torch.arange(self.max_seq_length, device=lengths.device).unsqueeze(0)  # shape [1, max_len]
            mask = mask.repeat(batch_size, 1)  # [batch_size, max_len]
            # mask[i, j] = True if j < lengths[i], else False
            valid_mask = (mask < lengths.unsqueeze(1)).to(lstm_out.device)  # [batch_size, max_len] bool

            # (2.3) 計算 dot-product scores = (lstm_out * query)
            #       先把 query shape 改成 [batch_size, 1, hidden_size]
            query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
            # print("lstm out", lstm_out.shape)
            # print("query", query.shape)
            # element-wise multiply & sum over hidden_size => scores.shape = [batch_size, max_seq_len]
            scores = (lstm_out * query).sum(dim=2)  # dot product

            # 把 padding 部分的 scores 設成非常小 => 不會被 softmax 選到
            scores = scores.masked_fill(~valid_mask, float('-inf'))

            # attn weight
            attn_weights = torch.nn.functional.softmax(scores, dim=1)  # [batch_size, max_seq_len]

            # (2.6) 加權求和 => context
            # broadast => [batch_size, max_seq_len, 1]
            attn_weights = attn_weights.unsqueeze(2)
            # lstm_out: [batch_size, max_seq_len, hidden_size]
            context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, hidden_size]

            # (optional) hidden layer
            context = self.hidden_layer(context)

            # =========== out layer ============
            out = self.output_layer(context)  # [batch_size, sum(action_space.nvec)]

            # (B) split => e.g. [3,3,3]
            action_logits = torch.split(out, self.action_space.nvec.tolist(),
                                        dim=-1)

            # 組合成 [batch_size, action_dims, num_classes]
            # 這裡 action_dims=3, num_classes=3 => shape [batch_size, 3, 3]
            stacked = torch.stack(action_logits, dim=1)
            # print("stacked", stacked.shape)
            return stacked

    def predict(self, x, reset=False):

        if reset:
            self._reset_to_initial_state()
        self.eval()
        with torch.no_grad():
            out = self.forward(x, testing_mode=True, num_tau=self.K)
            out = out.cpu().numpy().squeeze()
        return out

    def _reset_to_initial_state(self):
        self.sequence_buffer = []

