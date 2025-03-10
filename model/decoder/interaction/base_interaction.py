from torch import nn

class BaseInteraction(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config

    def forward(self, hidden1, hidden2):
        NotImplementedError("no implemented")

#import torch.nn as nn
from common.utils import HiddenData

class BaseInteraction1(nn.Module):
    def __init__(self, hidden_dim=768, dropout_rate=0.1):
        super(BaseInteraction, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Attention layer to enhance intent-slot interactions
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout_rate)

        # Gating mechanism (optional)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hiddens: HiddenData):
        """
        Forward pass to process and enhance hidden representations.
        """
        intent_hidden = hiddens.intent_hidden_state  # Intent hidden states
        slot_hidden = hiddens.slot_hidden_state      # Slot hidden states

        # Self-attention for interaction
        attn_output, _ = self.attn(slot_hidden, intent_hidden, intent_hidden)

        # Gating mechanism to merge information
        combined = torch.cat([slot_hidden, attn_output], dim=-1)
        gated_output = torch.tanh(self.gate(combined))

        # Update hidden states
        hiddens.update_slot_hidden_state(self.dropout(gated_output))
        hiddens.update_intent_hidden_state(self.dropout(intent_hidden))

        return hiddens


