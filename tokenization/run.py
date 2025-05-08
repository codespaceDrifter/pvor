import os
import sys
import json

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(projectRoot)

from usb import usb_path
from tokenization.tokenizer import Tokenizer
from tokenization.trie import Trie, count_pairs, add_top_pairs_to_dict, base_dict
import string

# --- Config ---
script_dir = os.path.dirname(__file__)
SAVE_FILE = os.path.join(script_dir, "assets", "token_to_id.json")

token_dict = base_dict.copy()

# --- BPE Training Loop ---
    
def train_tokenizer(input_file,
                    save_file,
                    token_dict,
                    target_vocab_size,
                    lines_to_read,
                    merge_per_step):

    while len(token_dict) < target_vocab_size:
        print ("token dict len", len(token_dict))

        counter = count_pairs (input_file, token_dict, lines_to_read)
        token_dict = add_top_pairs_to_dict(counter, token_dict, merge_per_step, save_file)

GOOGLE_INPUT_FILE = os.path.join(script_dir, "../dataset/20kSpaced.txt")

GOOGLE_TARGET = 12_000
GOOGLE_LINES = 20_000
GOOGLE_MERGE = 20

train_tokenizer(GOOGLE_INPUT_FILE, SAVE_FILE, token_dict, GOOGLE_TARGET, GOOGLE_LINES, GOOGLE_MERGE)


WEB_INPUT_FILE = os.path.join(usb_path(), "webtext", "openwebtext.txt")
TARGET_VOCAB_SIZE = 16_000
WEB_LINES = 50_000
WEB_MERGE = 50


train_tokenizer(WEB_INPUT_FILE, SAVE_FILE, token_dict, TARGET_VOCAB_SIZE, WEB_LINES, WEB_MERGE)
    



