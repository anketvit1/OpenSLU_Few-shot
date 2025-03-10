import torch
import torch.nn as nn
from model.decoder.interaction.base_interaction import BaseInteraction
from common.utils import HiddenData  # Ensure correct import

class NewInteraction(BaseInteraction):
    def __init__(self, **config):
        """
        Custom Interaction Module for SLU.
        Processes hidden states from RoBERTa.
        """
        super(NewInteraction, self).__init__()
        self.config = config
        self.output_dim = config.get("output_dim", 768)
        self.dropout = nn.Dropout(config.get("dropout_rate", 0.1))
        self.fc = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, hiddens: HiddenData):
        """
        Processes RoBERTa hidden states and updates slot & intent hidden states.
        """
        # Debugging: Print available attributes
        print("HiddenData Attributes:", vars(hiddens))

        # 🔍 Use intent_hidden or slot_hidden as RoBERTa hidden states
        if hasattr(hiddens, "intent_hidden"):
            hidden_states = hiddens.intent_hidden  # ✅ Use intent hidden states
        elif hasattr(hiddens, "slot_hidden"):
            hidden_states = hiddens.slot_hidden  # ✅ Alternative slot hidden
        else:
            raise AttributeError("HiddenData does not contain intent_hidden or slot_hidden.")

        # Transform hidden states
        transformed_hidden = self.dropout(self.fc(hidden_states))

        # Update HiddenData
        hiddens.update_slot_hidden_state(transformed_hidden)
        hiddens.update_intent_hidden_state(transformed_hidden)

        return hiddens

