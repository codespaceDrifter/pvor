import torch
import torch.nn as nn
from model.basic.mask import create_mha_padding_mask, create_mha_causal_mask
from model.transformer.pos_encode import PositionalEncoding
from model.basic.ffw import FeedForward
from model.transformer.mha import MultiHeadAttention
from model.transformer.transformer_encoder import TransformerEncoder
from model.transformer.transformer_decoder import TransformerDecoder


class ClassicTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model,
                 num_heads,
                 num_encoders,
                 num_decoders,
                 d_ff, 
                 loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction = "mean"),
                 max_seq_len = 50000,
                 pad_id = 0,
                 sos_id = 1,
                 eos_id = 2,
                 unk_id = 3,
                 dropout=0.1,):
        super().__init__()
        
        self.loss_fn = loss_fn
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encode = PositionalEncoding(d_model, max_seq_len)

        # Create encoder layers
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                d_model,
                MultiHeadAttention(d_model, num_heads),
                FeedForward(d_model, d_ff),
                dropout
            )
            for _ in range(num_encoders)
        ])
        
        # Create decoder layers
        self.decoders = nn.ModuleList([
            TransformerDecoder(
                d_model,
                MultiHeadAttention(d_model, num_heads),
                MultiHeadAttention(d_model, num_heads),
                FeedForward(d_model, d_ff),
                dropout
            )
            for _ in range(num_decoders)
        ])

        self.final_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        src_pad_mask = create_mha_padding_mask(src)
        tgt_pad_mask = create_mha_padding_mask(tgt)
        tgt_causal_mask = create_mha_causal_mask(tgt)
        tgt_combined_mask = tgt_pad_mask & tgt_causal_mask

        src_pad_mask = src_pad_mask.to(src.device)
        tgt_combined_mask = tgt_combined_mask.to(src.device)

        # Convert input IDs to embeddings
        src = self.embedding(src)

        tgt = self.embedding(tgt)
        
        # Add positional encoding
        src = self.pos_encode(src)
        tgt = self.pos_encode(tgt)

        for encoder in self.encoders:
            src = encoder(src, src_pad_mask)

        for decoder in self.decoders:
            tgt = decoder(tgt, src, tgt_combined_mask, src_pad_mask)



            
        # Convert back to vocabulary size
        output = self.final_layer(tgt)
        return output
    
    def compute_loss(self, src, tgt):
        assert torch.isfinite(src).all(), "NaN in src"
        assert torch.isfinite(tgt).all(), "NaN in tgt"
        
        outputs = self.forward(src, tgt)

        assert torch.isfinite(outputs).all(), "NaN in output"
         
                
        outputs = outputs[:, :-1]
        #(batch, vocab,seq_len)
        outputs = outputs.permute(0, 2, 1)
        #tgt shape is (batch, seq_len)

        # set UNK id to PAD id to ignore UNK. since PAD is already ignored. 
        #use .clone() to not modify original version other wise autograd gets broken
        tgt = tgt.clone()
        tgt[tgt == self.unk_id] = self.pad_id

        assert tgt.max().item() < self.vocab_size and tgt.min() >= 0,, "tgt out of range"

        '''
        Cross-Entropy Derivation (from softmax to logit form):
        For the true class y:
        p_y = e^{z_y} / sum_j e^{z_j}

        So cross entropy becomes, with z as logits:
        CE = -log(p_y)
        CE = -log(e^{z_y} / sum_j e^{z_j})
        CE = -z_y + log(sum_j e^{z_j})
        '''
        
        loss = self.loss_fn(outputs, tgt[:, 1:].long())

        assert torch.isfinite(loss).all(), "NaN in loss"
        
        return loss


    def predict(self, src, tgt):
        output = self.forward(src, tgt)
        pred_probs = output[:,-1]
        pred_id = pred_probs.argmax(dim=-1)
        pred_id = pred_id.unsqueeze(-1)
        return pred_id




