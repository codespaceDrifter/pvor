import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
DATA_path = os.path.abspath(os.path.join(project_root, '../DATA'))

from model.transformer.classic_transformer import ClassicTransformer
import torch.nn as nn
import torch

from utils.unstructured_text_dataset import TransformerDataset



train_dataset = TransformerDataset(
    path=os.path.join(DATA_path, "train_part_aa.bin"),
    input_len=128,
    output_len=128,
    stride=256,
    dtype=torch.int32
)

test_dataset = TransformerDataset(
    path=os.path.join(DATA_path, "test.bin"),
    input_len=128,
    output_len=128,
    stride=256,
    dtype=torch.int32)

print(f"Train dataset length: {len(train_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")
print(f"Train dataset sample: {train_dataset[0]}")

from tokenization.tokenizer import Tokenizer

tokenizer = Tokenizer(os.path.join(project_root, "tokenization/assets/token_to_id.json"))

first_20 = {k: tokenizer.id_to_token[k] for k in list(tokenizer.id_to_token.keys())[:20]}
print(first_20)

vocab_size = len (tokenizer.id_to_token)
print ("vocab size ", vocab_size)

model = ClassicTransformer(
    vocab_size = vocab_size,
    d_model = 1024, 
    num_heads = 16,
    num_encoders = 8,
    num_decoders = 8,
    d_ff = 4096,
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction = "mean"),
    max_seq_len = 10000,
    pad_id = 0,
    sos_id = 1,
    eos_id = 2,
    unk_id = 3,
    dropout=0.1
)

print (model)


parameter_num = sum(p.numel() for p in model.parameters() )
# 385 M
print(f"parameters num: {parameter_num:,}")




from train.trainer import train_model

train_model(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
    batch_size=24,
    save_folder_path=os.path.join(project_root, "checkpoints"),
    perma_save_folder_path=os.path.join(project_root, "checkpoints/perma"),
    loss_fn=nn.CrossEntropyLoss(ignore_index=0),
    tokenizer=tokenizer,
    batch_per_save=100,
    clip_grad_norm = 5
)


