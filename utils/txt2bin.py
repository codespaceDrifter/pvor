import torch
import os
import re

import os
import json
import sys

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(projectRoot)

from tokenization.tokenizer import Tokenizer
from usb import usb_path



def normalize_unicode_punctuation(text):
    replacements = {
        '‘': "'",  # left single quote
        '’': "'",  # right single quote
        '“': '"',  # left double quote
        '”': '"',  # right double quote
        '–': '-',  # en dash
        '—': '-',  # em dash
        '…': '...',  # ellipsis
        '•': '*',  # bullet
        '´': "'",  # acute accent used as apostrophe
        '″': '"',  # double prime
        '‹': '<', '›': '>',  # angle quotes
        '«': '<<', '»': '>>',
    }

    # Replace each unicode char with its ascii equivalent
    for uni, ascii_rep in replacements.items():
        text = text.replace(uni, ascii_rep)

    # Optional: remove any leftover weird control characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    return text



def tokenize_to_dataset(tokenizer: Tokenizer,
                        input_path: str,
                        train_path: str,
                        test_path: str,
                        valid_path: str,
                        train_ratio: float = 0.8,
                        test_ratio: float = 0.1,
                        valid_ratio: float = 0.1,
                        start_line: int = 0,
                        dtype: torch.dtype = torch.int32):
    assert abs(train_ratio + test_ratio + valid_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First pass: just count lines
    print("Counting lines...")
    total_lines = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                total_lines += 1
    
    train_end = int(total_lines * train_ratio)
    test_end = train_end + int(total_lines * test_ratio)
    
    # Second pass: process one line at a time
    processed_lines = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as in_file, \
         open(train_path, "ab") as train_file, \
         open(test_path, "ab") as test_file, \
         open(valid_path, "ab") as valid_file:
        
        for line in in_file:
            line = line.strip()
            line = normalize_unicode_punctuation(line)

            if not line:
                continue

            if processed_lines < start_line:
                processed_lines += 1
                continue

                
            # Process single line
            ids = tokenizer.encode(line, add_SOS=False, add_EOS=False).tolist()
            arr = torch.tensor(ids, dtype=dtype).numpy().tobytes()
            
            # Write to appropriate split
            if processed_lines < train_end:
                train_file.write(arr)
            elif processed_lines < test_end:
                test_file.write(arr)
            else:
                valid_file.write(arr)
                
            processed_lines += 1
            if processed_lines % 1000 == 0:
                print(f"Processed {processed_lines}/{total_lines} lines ({processed_lines/total_lines*100:.2f}%)")
                print(f"Line: {line}")
                print(f"IDs: {ids}")
        
        print(f"✅ Done. Split into {train_end} train, {test_end-train_end} test, {total_lines-test_end} valid examples")




tokenizer = Tokenizer(os.path.join(projectRoot, "tokenization/assets/token_to_id.json"))
input_path = os.path.join(usb_path(), "webtext", "openwebtext.txt")
train_path = os.path.join(usb_path(), "webtext", "train.bin")
test_path= os.path.join(usb_path(), "webtext", "test.bin")
vlad_path = os.path.join(usb_path(), "webtext", "vlad.bin")

tokenize_to_dataset(tokenizer = tokenizer,
                        input_path= input_path,
                        train_path= train_path,
                        test_path=test_path,
                        valid_path= vlad_path)
