import regex as re

def train_bpe(
        #input_path : str,
        #vocab_size : int,
        #spetial_tokens : list[str],
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Args:
        input_path: str Path to a text file with BPE tokenizer training data.
        vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.

    Returns:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    """
    # GPT-2 regex-based pre-tokenizer
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    patterns = re.finditer(PAT, 
                        """low low low low low lower lower widest widest widest newest newest newest newest newest newest""")
    p_frequency = {}
    b_vocab = {}

    for pattern in patterns:
        btuple_pattern = tuple(bytes(s, 'utf-8') for s in pattern.group())
        if not p_frequency.get(btuple_pattern, None):
            p_frequency[btuple_pattern] = 1
        else:
            p_frequency[btuple_pattern] += 1
    for btuple in p_frequency.keys():
        for i in range(len(btuple) - 1):
            pair = bytes(btuple[i].decode() + btuple[i + 1].decode(), 'utf-8')
            
            if not b_vocab.get(pair, None):
                b_vocab[pair] = 1
            else:
                b_vocab[pair] += 1

    
    max_frequency = max(b_vocab.values())
    greater_pair = max([key for key in b_vocab.keys() if b_vocab[key] == max_frequency])
    
    for btuple in p_frequency.copy().keys():
        new_btuple = []

        for i in range(len(btuple) - 1):
            pair = bytes(btuple[i].decode() + btuple[i + 1].decode(), 'utf-8')
            if pair.decode() == greater_pair.decode():
                new_btuple.append(pair)
            elif i == len(btuple) - 1:
                new_btuple.append(btuple[i + 1])
            else:
                new_btuple.append(btuple[i])
        p_frequency[tuple(new_btuple)] = p_frequency[btuple]
        p_frequency.pop(btuple)

    print(p_frequency)
    


train_bpe()
