from model.decoder.base_decoder import BaseDecoder
from model.decoder.classifier import LinearClassifier
from common.utils import HiddenData
from common.utils import OutputData
class NewDecoder(BaseDecoder):
    def __init__(self, interaction, slot_cls, intent_cls):
        super(NewDecoder, self).__init__()
        self.interaction = interaction
        self.slot_cls = slot_cls  # Slot classification layer
        self.intent_cls = intent_cls  # Intent classification layer

    def forward(self, hiddens):
        # Interaction processing
        interact = self.interaction(hiddens)  

        # 🔧 Fix: Use `slot_hidden` instead of `slot`
        slot_output = self.slot_cls(interact.slot_hidden)  # ✅ Corrected

        # Intent classification
        intent_output = self.intent_cls(interact.intent_hidden)

        return slot_output, intent_output

