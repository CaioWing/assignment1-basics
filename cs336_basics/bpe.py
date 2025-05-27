from collections import Counter, defaultdict
from itertools import pairwise
import mmap
import regex as re
import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boudaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def count_chunk(chunk: bytes, SPLIT_RE: re.Pattern, PAT_RE: re.Pattern) -> Counter:
    """
    Count the words of a chunk of text
    """
    local = Counter()
    for doc in SPLIT_RE.split(chunk):
        local.update(PAT_RE.findall(doc))   
    return local

def count_words(path: str, SPLIT_RE: re.Pattern, PAT_RE: re.Pattern) -> Counter:
    """
    Count the words of a file
    """
    word_counts = Counter()
    with open(path, "rb") as f, \
         mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:

        boundaries = find_chunk_boundaries(mm, os.cpu_count(), SPLIT_RE.pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            word_counts += count_chunk(mm[start:end], SPLIT_RE, PAT_RE)
    return word_counts

def train_bpe(
        input_path : str,
        vocab_size : int,
        special_tokens : list[str],
        chunk_size: int = 10
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

    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    merges = []
    word_counts = Counter()

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    special_token_pattern = "|".join([re.escape(token) for token in special_tokens])
    PAT_RE   = re.compile(PAT.encode("utf-8"))
    SPLIT_RE = re.compile(special_token_pattern.encode("utf-8"))

    word_counts = count_words(input_path, SPLIT_RE, PAT_RE)
    word_counts = {tuple(bytes([b]) for b in k): v for k, v in word_counts.items()}

    while len(vocab) < vocab_size:
        pair_freq = defaultdict(int)

        for word_tuple, freq in word_counts.items():
            for pair in pairwise(word_tuple):  
                pair_freq[pair] += freq 

        if not pair_freq:
            break

        max_freq = max(pair_freq.values())
        best_pair = max([pair for pair, freq in pair_freq.items() if freq == max_freq])

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1

        new_word_freq = {}
        for word_tuple, freq in word_counts.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_word_freq[tuple(new_word)] = freq
        word_counts = new_word_freq
    return vocab, merges 

if __name__ == "__main__":
    from scalene import scalene_profiler
    scalene_profiler.start()
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 500, ["<|endoftext|>"])
    scalene_profiler.stop()