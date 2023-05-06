import torch.nn as nn


class RewardTarget(nn.Module):
    """
    RM Target for RLHF
    """
    def __init__(self, args, vocab_size):
        super(RewardTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size

        self.linear_1 = nn.Linear(args.hidden_size, 1)

    def forward(self, memory_bank, tgt=None, seg=None):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length] -> in this target, we use for attention mask.

        Returns:
            loss: Classification loss.
            correct: Number of sentences that are predicted correctly.
        """
        # output = self.linear_1(memory_bank).reshape(-1, memory_bank.shape[1])
        # seg = seg.sum(dim=1).reshape(-1, 1).long() - 1
        # loss = torch.gather(output, dim=1, index=seg)

        # output = self.linear_1(memory_bank).reshape(-1, memory_bank.shape[1])
        # loss = (output * seg).sum(dim=1) / seg.sum(dim=1)
        logits = self.linear_1(memory_bank).squeeze(-1)

        return logits
