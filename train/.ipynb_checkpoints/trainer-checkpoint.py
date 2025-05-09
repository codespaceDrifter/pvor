import os
import sys

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(projectRoot)

import torch
import os
from torch.utils.data import DataLoader
from tokenization.tokenizer import Tokenizer
from train.saves import cleanup_checkpoints, load_latest_checkpoint
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import time





'''
            outputs = model(inputs,targets[:, :-1])
            outputs = outputs.permute(0, 2, 1)
            loss = loss_fn(outputs, targets[:, 1:].long())
these 3 lines basically. the forward_pass function

'''

def train_model(
    train_dataset,
    test_dataset,
    model,
    optimizer,
    batch_size,
    save_folder_path,
    perma_save_folder_path,
    loss_fn,
    tokenizer = None,
    batch_per_save = 10,
    clip_grad_norm = 2.0
):
    # to debug
    # ! COMMENT THIS OUT IN ACTUAL TRAINING. MAKES IT 2 TIMES SLOWER
    #torch.autograd.set_detect_anomaly(True)


    #mix precision training
    scaler = GradScaler()

    model.cuda()
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_folder_path), exist_ok=True)
    
    # Try to load latest checkpoint and get last batch
    last_batch = load_latest_checkpoint(save_folder_path, model)




    

    batch_idx = last_batch

    perma_save = max (batch_per_save * 10, 500)
    
    # Training loop
    model.train()
    
            
    for inputs, targets in train_loader:

        batch_idx += 1
                
        inputs = inputs.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()

        #mixed precision about a 30% speed up
        with autocast():
            loss = model.compute_loss(inputs,targets)
        scaler.scale(loss).backward()

        #unscales. i.e. divide by 1024 which scale multipled by 1024. to clip grad norm. if delete this line, scaler.step will also unscale
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        

        

            
    # Print loss every 10 batches
        if (batch_idx) % batch_per_save == 0:
            current_loss = loss.item()
            print(f"Batch {batch_idx}, Train Loss: {current_loss:.4f}")

    # save every 10 batches
        if (batch_idx) % batch_per_save == 0:
            state = model.state_dict()
            torch.save(state, f"{save_folder_path}/batch_{batch_idx}.pt")
            cleanup_checkpoints(save_folder_path)

    #perma save every save_batch * 10 batches. permanent save do not delete
        if (batch_idx) % perma_save == 0:
            state = model.state_dict()
            torch.save(state, f"{perma_save_folder_path}/batch_{batch_idx}.pt")
            cleanup_checkpoints(perma_save_folder_path)

    # eval one test batch every 10 batches
        if (batch_idx) % batch_per_save == 0:
            model.eval()
            with torch.no_grad():
                try:
                    test_inputs, test_targets = next(iter(test_loader))


                    test_inputs = test_inputs.cuda()
                    test_targets = test_targets.cuda()

                    test_outputs = model(test_inputs, test_targets[:, :-1])
                    permuted_outputs = test_outputs.permute(0, 2, 1)
                    test_loss = loss_fn(permuted_outputs, test_targets[:, 1:].long()).item()
                    print(f"Test Loss: {test_loss:.4f}")

                    print("Target:")
                    target_text = tokenizer.decode(test_targets[0].cpu())
                    print(target_text)
                            
                    print("Output:")
                    pred_id = test_outputs.argmax(dim=-1)

                    output_text = tokenizer.decode(pred_id[0].cpu())
                    print(output_text)

                except StopIteration:
                    print("No more test batches")
                    break


# nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv --loop=1
