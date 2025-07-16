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
        self.SPECIAL_TOKEN_PAT= re.compile("|".join([re.escape(token) for token in special_tokens])) if special_tokens else None   

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
        if self.SPECIAL_TOKEN_PAT:
            text = self.SPECIAL_TOKEN_PAT.split(text)

        words = self.PRE_TOKENIZATION.findall(text)
        encode = []

        for word in words:
            word = word.replace(' ', self.SPACE_CHAR)
            new_word = []
            idx = 0
            
            while True:
                while idx < len(word):
                    if idx < len(word) - 1 and (word[idx], word[idx + 1]) in self.merges:
                        new_word.append((word[idx] + word[idx + 1]))
                        idx += 2
                    else:
                        new_word.append(word[idx])
                        idx += 1
                if word == tuple(new_word):
                    encode.extend([encoding_vocab[token.encode()] for token in new_word])
                    break
                else:
                    word = tuple(new_word.copy())
        return encode
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return 
        a generator that lazily yields token IDs. This is required for 
        memory-efficient tokenization of large files that we cannot 
        directly load into memory.
        """
        return "".join([self.encode(out) for out in iterable])
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """
        return "".join([self.vocab[i].decode() for i in ids]).replace(self.SPACE_CHAR, " ")

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