"""
gradcam.py — Boilerplate for Grad-CAM on EEG Parallel CNN-GRU.

This class handles hook registration and data capture. 
The core mathematical logic for generating the heatmap is left for you!
"""

import torch
import torch.nn as nn
import numpy as np


class EEGGradCAM:
    """
    Base class for generating Grad-CAM heatmaps for a CNN layer.
    
    Usage:
        gcam = EEGGradCAM(model, target_layer=model.cnn.conv_block3)
        heatmap = gcam.generate(input_cnn, input_rnn, target_class=1)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        
        # Placeholders for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Setup forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Registering the hooks to the target_layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, cnn_x: torch.Tensor, rnn_x: torch.Tensor, target_class: int = None):
        """
        Generates the Grad-CAM heatmap.
        
        Args:
            cnn_x        : CNN input tensor [1, 1, channels, time]
            rnn_x        : RNN input tensor [1, time, features]
            target_class : The index of the class to explain. 
                          If None, uses the model's predicted class.
        
        Returns:
            heatmap : A 2D numpy array [channels, time] (or whatever the layer shape is)
        """
        # 1. Forward pass
        logits = self.model(cnn_x, rnn_x)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # 2. Backward pass
        self.model.zero_grad()
        # Goal: calculate gradients of the target class score w.r.t. feature maps
        score = logits[0, target_class]
        score.backward()

        # ---------------------------------------------------------------------
        # YOUR TASK STARTS HERE
        # ---------------------------------------------------------------------
        
        # Access captured data:
        # self.activations: [batch, filters, H, W] (e.g. [1, 32, 64, 10])
        # self.gradients:   [batch, filters, H, W]
        
        # 1. Calculate weights (global average pool the gradients)
        # weights = ...

        # 2. Weighted sum of activations
        # heatmap = ...

        # 3. Apply ReLU to keep only positive contributions
        # heatmap = ...

        # 4. Normalise the result between 0 and 1
        # heatmap = ...

        # ---------------------------------------------------------------------
        # YOUR TASK ENDS HERE
        # ---------------------------------------------------------------------

        return None # Return your processed heatmap here
