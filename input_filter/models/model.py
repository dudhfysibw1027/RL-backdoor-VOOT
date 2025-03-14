import torch
from torch import nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32, momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64, momentum=0.01),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # 🔹 計算 CNN 最後輸出的 seq_len
        configs.features_len = self._get_final_seq_len(configs.input_channels, configs.seq_len)

        # 🔹 設定 Linear 層
        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)

    def _get_final_seq_len(self, input_channels, seq_len):
        """ 計算 CNN 之後的輸出序列長度 """
        x = torch.zeros(1, input_channels, seq_len)  # 創建假輸入
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x.shape[-1]  # CNN 最終輸出的 seq_len

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)  # (batch_size, features_len * final_out_channels)
        logits = self.logits(x_flat)
        return logits, x
