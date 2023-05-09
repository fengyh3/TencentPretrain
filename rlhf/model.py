import torch.nn as nn


class Model(nn.Module):
    """
    Pretraining models consist of three (five) parts:
        - embedding
        - encoder
        - tgt_embedding (optional)
        - decoder (optional)
        - target
    """

    def __init__(self, args, target_name, embedding, encoder, tgt_embedding, decoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.tgt_embedding = tgt_embedding
        self.decoder = decoder
        self.target = target

        if "mlm" in target_name and args.tie_weights:
            self.target.mlm.linear_2.weight = self.embedding.word.embedding.weight
        elif "lm" in target_name and args.tie_weights and "word" in self.embedding.embedding_name_list:
            self.target.lm.output_layer.weight = self.embedding.word.embedding.weight
        elif "lm" in target_name and args.tie_weights and "word" in self.tgt_embedding.embedding_name_list:
            self.target.lm.output_layer.weight = self.tgt_embedding.word.embedding.weight

        if self.decoder is not None and args.share_embedding:
            self.tgt_embedding.word.embedding.weight = self.embedding.word.embedding.weight

    def forward(self, src, tgt, seg, tgt_in=None, tgt_seg=None, return_logits=False):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        if self.decoder:
            tgt_emb = self.tgt_embedding(tgt_in, tgt_seg)
            memory_bank = self.decoder(memory_bank, tgt_emb, (seg, tgt_seg))

        if return_logits and hasattr(self.target, "lm"):
            loss_info = self.target.lm.output_layer(memory_bank)
            loss_info = nn.LogSoftmax(dim=-1)(loss_info)
        else:
            if tgt_seg is not None:
                loss_info = self.target(memory_bank, tgt, tgt_seg)
            else:
                loss_info = self.target(memory_bank, tgt, seg)

        return loss_info
