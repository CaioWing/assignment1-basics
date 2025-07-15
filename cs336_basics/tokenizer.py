from collections.abc import Iterable
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
        self.PRE_TOKENIZATION = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.SPECIAL_TOKEN_PAT= re.compile("|".join([re.escape(token) for token in special_tokens])) if special_tokens else None   

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
            vocab = {}
            for line in f:
                line_split = line.decode('utf-8').split(": b")
                key = line_split[0]
                token_delimiter = line_split[-1][0]
                token = line_split[-1][1:].rstrip(f"{token_delimiter}\n")
                try:
                    token = token.encode('utf-8').decode('unicode_escape').encode('utf-8')
                except:
                    token = token.encode('utf-8')
                vocab[int(key)] = bytes(token)
            
        with open(merges_filepath) as f:
            merges = []
            for line in f:
                breakpoint()
                matches = [m.encode('utf8') for m in re.findall(r"b'([^']*)'", line)]
                breakpoint()
                merges.extend([tuple(bytes(matches[0]), bytes(matches[1]))])
        breakpoint()
        return cls.__init__(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        encoding_vocab = {v: k for k, v in self.vocab.items()}
        if self.SPECIAL_TOKEN_PAT:
            text = self.SPECIAL_TOKEN_PAT.split(text)

        words = self.PRE_TOKENIZATION.findall(text)
        byte_words = [tuple(letter.encode("utf-8") for letter in word) for word in words]
        encode = []

        for word in byte_words:
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
                    encode.extend([encoding_vocab[token] for token in new_word])
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
        return NotImplemented
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """
        return NotImplemented

if __name__ == "__main__":
    from cs336_basics.train_bpe import train_bpe

    # vocab, merges = train_bpe(input_path="data/TinyStoriesV2-GPT4-valid.txt", vocab_size=1000, special_tokens=["<|endoftext|>"])
    # Tokenizer(vocab, merges).encode("That's a time when I was young that everything was different, I dont recognize anymore")
    tokenizer = Tokenizer.from_files(Tokenizer, "vocab.txt", "merges.txt")
    breakpoint()