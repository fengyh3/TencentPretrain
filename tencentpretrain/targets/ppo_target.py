import torch
import torch.nn as nn

from tencentpretrain.utils.constants import *


class PPOTarget(nn.Module):
    """
    Language Model Target
    """

    def __init__(self, args, vocab_size):
        super(PPOTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        if "label_smoothing" in args:
            self.label_smoothing = args.label_smoothing
        else:
            self.label_smoothing = None
        if "ignore_index" in args and args.ignore_index:
            self.ignore_index = args.tokenizer.vocab.get(PAD_TOKEN)
        else:
            self.ignore_index = None
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size, bias=args.has_lmtarget_bias)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def lm(self, memory_bank, seg):
        # Language modeling (LM) with full softmax prediction.

        # seg = seg.contiguous().view(-1)
        # memory_bank = memory_bank.contiguous().view(-1, self.hidden_size)
        # memory_bank = memory_bank[seg > 0, :]
        output = self.output_layer(memory_bank)
        output = self.softmax(output)

        return output

    def forward(self, memory_bank, tgt, seg):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            logits: [batch_size x seq_length x vocab_size]
        """
        # Language modeling (LM) with full softmax prediction.
        logits = self.lm(memory_bank, seg)

        return logits