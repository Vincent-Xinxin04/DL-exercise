import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.downsample(x)
        return F.gelu(out)

class BinaryVulnModel(nn.Module):
    def __init__(self, byte_vocab_size=256, embedding_dim=128, meta_dim=10):
        super(BinaryVulnModel, self).__init__()
        
        # Byte Sequence Encoder
        self.embedding = nn.Embedding(byte_vocab_size + 1, embedding_dim)
        
        # Deep ResNet-1D for hierarchical feature extraction
        self.byte_encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ResBlock1D(128, 128),
            ResBlock1D(128, 256, stride=2),
            ResBlock1D(256, 256),
            ResBlock1D(256, 512, stride=2),
            ResBlock1D(512, 512)
        )
        self.adaptive_max = nn.AdaptiveMaxPool1d(1)
        self.adaptive_avg = nn.AdaptiveAvgPool1d(1)
        
        # Metadata Encoder (MLP)
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        
        # Combined Head (1024 from ResNet + 128 from Meta)
        self.fc_combined = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Dual Task Heads
        self.detection_head = nn.Linear(256, 2)
        self.classification_head = nn.Linear(256, 86)

    def forward(self, byte_seq, meta_features):
        # byte_seq: [batch_size, seq_len]
        x = self.embedding(byte_seq).transpose(1, 2) # [B, C, L]
        x = self.byte_encoder(x) # [B, 512, L']
        
        x_max = self.adaptive_max(x).squeeze(-1)
        x_avg = self.adaptive_avg(x).squeeze(-1)
        x = torch.cat([x_max, x_avg], dim=1) # [B, 1024]
        
        m = self.meta_fc(meta_features) # [B, 128]
        
        combined = torch.cat([x, m], dim=1)
        combined = self.fc_combined(combined)
        
        det_out = self.detection_head(combined)
        class_out = self.classification_head(combined)
        
        return det_out, class_out
