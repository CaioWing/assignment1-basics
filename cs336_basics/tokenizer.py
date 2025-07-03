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
        return NotImplemented

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        i = 0
        if self.SPECIAL_TOKEN_PAT:
            text = self.SPECIAL_TOKEN_PAT.split(text)

        words = self.PRE_TOKENIZATION.findall(text)
        byte_words = [tuple(letter.encode("utf-8") for letter in word) for word in words]
        
        while i < len(self.merges):
            for index, word in enumerate(byte_words):
                new_word = []
                for j in range(len(word)):
                    if j < len(word) - 1 and (word[j], word[j + 1] == self.merges[i]):
                        new_word.append(merges[i])
                        i = 0
                    else:
                        new_word.append(word[j])
                        i += 1
                byte_words[index] = tuple(new_word)
                breakpoint()
                
            
        return NotImplemented
    
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

    vocab, merges = train_bpe(input_path="data/TinyStoriesV2-GPT4-valid.txt", vocab_size=1000, special_tokens=["<|endoftext|>"])
    Tokenizer(vocab, merges).encode("Ola meu nome e caio")
