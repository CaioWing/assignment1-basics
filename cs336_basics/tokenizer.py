from collections.abc import Iterable
import json
from typing import Iterator
import regex as re


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, 
        and (optionally) a list of special tokens. This function should 
        accept the following parameters:

            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        self.vocab            = vocab
        self.merges           = merges
        self.special_tokens   = special_tokens

        self.SPACE_CHAR = '\u0120'
        self.PRE_TOKENIZATION = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        if special_tokens:
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
            self.SPECIAL_TOKEN_PAT = re.compile(f"({'|'.join(escaped_tokens)})")
        else:
            self.SPECIAL_TOKEN_PAT = None        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized 
        vocabulary and list of merges (in the same format that your BPE training 
        code output) and (optionally) a list of special tokens. This method 
        should accept the following additional parameters:

            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "rb") as f:
            vocab = {v : k for v, k in json.load(f).items()}
        
        if special_tokens:
            idx = len(vocab)
            for s_t in special_tokens:
                vocab[s_t] = idx
                idx += 1

        with open(merges_filepath) as f:
            merges = []
            for line in f:
                token_1 = line.split(' ')[0]
                token_2 = line.split(' ')[-1].replace('\n', '')
                merges.extend([(token_1, token_2)])
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        encoding_vocab = {v: k for k, v in self.vocab.items()}
        special_tokens = set()
        encode = []
        
        if self.SPECIAL_TOKEN_PAT:
            special_tokens = set(self.SPECIAL_TOKEN_PAT.findall(text))
            result = [item for item in self.SPECIAL_TOKEN_PAT.split(text) if item]
        else:
            result = [text]

        for sentence in result:
            if sentence in special_tokens:
                encode.append(encoding_vocab[sentence.encode()])
                continue 

            for word in self.PRE_TOKENIZATION.findall(sentence):
                tokens = [bytes([b]) for b in word.encode()]
                
                # Apply all the possible merges
                while len(tokens) > 1:
                    merged = False
                    
                    # Run through all the merges in order
                    for merge_pair in self.merges:
                        # Look for a specific merge pair
                        for i in range(len(tokens) - 1):
                            if (tokens[i], tokens[i + 1]) == merge_pair:
                                # Apply merge
                                merged_token = tokens[i] + tokens[i + 1]
                                tokens = tokens[:i] + [merged_token] + tokens[i + 2:]
                                merged = True
                                break
                        
                        if merged:
                            break # If the merge is applied, just look for another one

                    # If any merge is avaiable, the str is already fully encoded
                    if not merged:
                        break

                for token in tokens:
                    encode.append(encoding_vocab[token])
        
        return encode
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return 
        a generator that lazily yields token IDs. This is required for 
        memory-efficient tokenization of large files that we cannot 
        directly load into memory.
        """
        encoding = []
        for out in iterable:
            encoding.extend(self.encode(out))
        return encoding
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """
        if len(ids) == 0:
            return ""

        idx = 0
        b_string = self.vocab[ids[idx]]
        idx += 1
        while idx < len(ids):
            b_string += self.vocab[ids[idx]]
            idx += 1
        return b_string.decode(errors='replace').replace(self.SPACE_CHAR, " ")

if __name__ == "__main__":
    from cs336_basics.train_bpe import train_bpe

    # vocab, merges = train_bpe(input_path="data/TinyStoriesV2-GPT4-valid.txt", vocab_size=1000, special_tokens=["<|endoftext|>"])
    # Tokenizer(vocab, merges).encode("That's a time when I was young that everything was different, I dont recognize anymore")
    tokenizer = Tokenizer.from_files( 
        "TinyStoriesV2-GPT4-train.vocab.json", 
        "TinyStoriesV2-GPT4-train.merges.txt"
        )
    
    encoding = tokenizer.encode("ay That's a time when I was young that everything was different, I dont recognize anymore")
    decoding = tokenizer.decode(encoding)
    breakpoint()