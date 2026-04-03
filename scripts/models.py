"""
models.py — PyTorch model definition for EEG Motor Imagery classification.

Architecture: ParallelCNNRNN
  CNN and LSTM branches run in parallel, fused before the classification head.
  Supports four fusion strategies: 'concat', 'add', 'concat_fc', 'concat_conv1d'.
"""

import torch
import torch.nn as nn


class ParallelCNNRNN(nn.Module):
    """Parallel CNN + LSTM with configurable fusion.

    Data flow:
        CNN branch: (B,W,1,H,Wd) → 2-D CNN per frame → sum over W → cnn_fc vector
        RNN branch: (B,W,n_electrodes) → linear projection → LSTM → rnn_fc vector
        Fusion    : configurable combination of the two branch outputs
        Output    : (B, n_classes) raw logits

    Fusion strategies (``fusion`` arg):
        'concat'       — simple concatenation, no extra parameters
        'add'          — element-wise addition (requires cnn_fc == rnn_fc_out)
        'concat_fc'    — concatenation → learnable FC layer
        'concat_conv1d'— concatenation → 1-D convolution (point-wise MLP)
    """

    def __init__(
        self,
        window_size: int = 10,
        conv_channels: int = 32,
        cnn_fc: int = 256,
        n_electrodes: int = 64,
        rnn_fc_in: int = 256,
        lstm_hidden: int = 16,
        lstm_layers: int = 2,
        rnn_fc_out: int = 128,
        n_classes: int = 4,
        dropout: float = 0.5,
        fusion: str = "concat",
        **kwargs,               # absorb extra config keys silently
    ):
        super().__init__()
        self.fusion = fusion
        ch = conv_channels

        # ── CNN branch ─────────────────────────────────────────────────────────
        self.cnn_features = nn.Sequential(
            nn.Conv2d(1,    ch,   3, padding=1), nn.BatchNorm2d(ch),   nn.ELU(inplace=True),
            nn.Conv2d(ch,   ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ELU(inplace=True),
            nn.Conv2d(ch*2, ch*4, 3, padding=1), nn.BatchNorm2d(ch*4), nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 4, cnn_fc),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── RNN branch ─────────────────────────────────────────────────────────
        self.rnn_proj = nn.Sequential(
            nn.Linear(n_electrodes, rnn_fc_in),
            nn.ELU(inplace=True),
        )
        self.lstm = nn.LSTM(
            rnn_fc_in, lstm_hidden, lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.rnn_head = nn.Sequential(
            nn.Linear(lstm_hidden, rnn_fc_out),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Fusion layer ────────────────────────────────────────────────────────
        if fusion == "concat":
            fused_dim = cnn_fc + rnn_fc_out
            self.fuse_layer = None
        elif fusion == "add":
            if cnn_fc != rnn_fc_out:
                raise ValueError(
                    f"'add' fusion requires cnn_fc == rnn_fc_out, "
                    f"got {cnn_fc} vs {rnn_fc_out}"
                )
            fused_dim = cnn_fc
            self.fuse_layer = None
        elif fusion == "concat_fc":
            fused_dim = cnn_fc + rnn_fc_out
            self.fuse_layer = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.ELU(inplace=True),
            )
        elif fusion == "concat_conv1d":
            fused_dim = cnn_fc + rnn_fc_out
            self.fuse_layer = nn.Sequential(
                nn.Conv1d(fused_dim, fused_dim, kernel_size=1),
                nn.ELU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion!r}")

        self.readout = nn.Linear(fused_dim, n_classes)

    def forward(self, cnn_x: torch.Tensor, rnn_x: torch.Tensor) -> torch.Tensor:
        B, W, C, H, Wd = cnn_x.shape

        # ── CNN branch ──
        cnn_frames = self.cnn_fc(
            self.cnn_features(cnn_x.reshape(B * W, C, H, Wd))
        )                                                          # (B*W, cnn_fc)
        cnn_out = cnn_frames.reshape(B, W, -1).sum(dim=1)        # sum over W → (B, cnn_fc)

        # ── RNN branch ──
        proj    = self.rnn_proj(rnn_x.reshape(B * W, -1)).reshape(B, W, -1)
        lstm_h  = self.lstm(proj)[0][:, -1, :]                   # last step → (B, lstm_hidden)
        rnn_out = self.rnn_head(lstm_h)                           # (B, rnn_fc_out)

        # ── Fusion ──
        if self.fusion == "concat":
            fused = torch.cat([cnn_out, rnn_out], dim=1)
        elif self.fusion == "add":
            fused = cnn_out + rnn_out
        elif self.fusion == "concat_fc":
            fused = self.fuse_layer(torch.cat([cnn_out, rnn_out], dim=1))
        elif self.fusion == "concat_conv1d":
            cat   = torch.cat([cnn_out, rnn_out], dim=1).unsqueeze(2)  # (B, fused_dim, 1)
            fused = self.fuse_layer(cat).squeeze(2)

        return self.readout(fused)


# ── Convenience factory ───────────────────────────────────────────────────────
def build_model(cfg: dict, window: int, n_classes: int, dropout: float) -> ParallelCNNRNN:
    """Instantiate ParallelCNNRNN from a config dict.

    Args:
        cfg       : model kwargs dict (e.g. config.PARALLEL_CFG)
        window    : window size (W)
        n_classes : number of output classes
        dropout   : dropout probability

    Returns:
        ParallelCNNRNN instance (not yet moved to a device)
    """
    return ParallelCNNRNN(window_size=window, n_classes=n_classes, dropout=dropout, **cfg)
