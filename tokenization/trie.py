import json
from collections import Counter, defaultdict
from pathlib import Path
import string

# === Initial vocab ===
BASE_TOKENS = ['PAD', 'SOS', 'EOS', 'UNK'] + list('0123456789abcdefghijklmnopqrstuvwxyz ') + list(string.punctuation)

base_dict = {tok: i for i, tok in enumerate(BASE_TOKENS)}

MAX_TOKEN_LENGTH = 20


# === Trie for fast longest-match ===
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_token = False

class Trie:
    def __init__(self, vocab):
        self.root = TrieNode()
        for token in vocab:
            self.insert(token)

    def insert(self, token):
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_token = True

    def tokenize(self, line):
        line = line.strip().lower()
        i = 0
        tokens = []
        while i < len(line):
            node = self.root
            longest_match = None
            j = i
            while j-i < MAX_TOKEN_LENGTH and j < len(line) and line[j] in node.children:
                node = node.children[line[j]]
                j += 1
                if node.is_token:
                    longest_match = line[i:j]
            if longest_match:
                tokens.append(longest_match)
                i += len(longest_match)
            else:
                tokens.append('UNK')
                i += 1
        return tokens


# === Step 1: Count token pairs from file ===
def count_pairs(file_path, vocab_dict, linesToRead):
    token_list = list(vocab_dict.keys()) 
    trie = Trie(token_list)
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in range (linesToRead):

            if (_ %1000 == 0): print ("reading line ", _)

            line = f.readline()
            if not line: break
            tokens = trie.tokenize(line)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                counter[pair] += 1
    return counter


# === Step 2: Add top N pairs to vocab and save ===
def add_top_pairs_to_dict(counter, token_to_id, N=100, save_path='bpe_vocab.json'):
    next_id = max(token_to_id.values()) + 1
    added = 0

    for (a, b), _ in counter.most_common():
        if added >= N:
            break

        # force space at the end
        if a == " ":
            a, b = b, " "
        
        #no unk in dicts
        if "UNK" in a or "UNK" in b:
            continue

        #keep digits pure
        if a.isdigit() or b.isdigit():
            continue

        new_token = a + b

        #prevent too many words being in one token
        if len(new_token) > MAX_TOKEN_LENGTH:
            continue

        if new_token not in token_to_id:
            token_to_id[new_token] = next_id
            next_id += 1
            added += 1  # only count if actually added

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(token_to_id, f, indent=2)

    return token_to_id
