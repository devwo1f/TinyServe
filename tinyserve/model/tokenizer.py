"""Tokenizer wrapper and streaming detokenizer.

The engine talks in token ids. Hugging Face `transformers` owns the Llama chat
template and the byte-level BPE itself (spec Section 5); this module is the
boundary so the rest of TinyServe does not depend on that API. Streaming
detokenization lives here because a token is not always a whole character.
"""

from typing import Any


class Tokenizer:
    """A Hugging Face tokenizer reduced to the operations the engine needs.

    `encode` does not add special tokens by default. The chat template already
    inserts the beginning-of-sequence token, and adding it twice would shift
    every position.
    """

    def __init__(self, backend: Any):
        self.backend = backend

    @classmethod
    def from_pretrained(cls, path: str) -> "Tokenizer":
        """Load a local checkpoint directory or a Hugging Face repo id.

        `clean_up_tokenization_spaces` is forced off: that cleanup is for
        WordPiece and deletes spaces before punctuation, which corrupts
        byte-level BPE text.
        """
        from transformers import AutoTokenizer

        backend = AutoTokenizer.from_pretrained(path, clean_up_tokenization_spaces=False)
        return cls(backend)

    @property
    def eos_token_id(self) -> int:
        """Id that ends a turn. Llama 3 Instruct uses `<|eot_id|>`, not the old end-of-text id."""
        eos = self.backend.eos_token_id
        if eos is None:
            raise ValueError("tokenizer has no eos token")
        return int(eos)

    @property
    def bos_token_id(self) -> int | None:
        bos = self.backend.bos_token_id
        return None if bos is None else int(bos)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Token ids for raw text. Special tokens stay off unless the caller asks."""
        return [int(i) for i in self.backend.encode(text, add_special_tokens=add_special_tokens)]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        """Text for token ids. Special tokens stay unless the caller is showing text to a user."""
        return self.backend.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Token ids for a chat, including the special tokens the template inserts.

        `add_generation_prompt` ends the prompt on the assistant header so the
        model writes the reply instead of seeing a closed turn.
        """
        encoded = self.backend.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(i) for i in ids]


class IncrementalDetokenizer:
    """Emit only the text that cannot change when later tokens arrive.

    Llama's tokenizer is byte-level BPE, so one Unicode character can be split
    across tokens (the emoji in "hello 🌟" is three tokens). Decoding the
    incomplete bytes yields U+FFFD, and the next token replaces that character.
    Sending the replacement to the client shows a box that then disappears, so
    trailing U+FFFD is held back until it resolves or `finish` is called.
    """

    def __init__(self, tokenizer: Tokenizer, *, skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.token_ids: list[int] = []
        self._emitted = ""

    def add(self, token_id: int) -> str:
        """Append one token and return the newly stable text. May be empty."""
        self.token_ids.append(int(token_id))
        return self._emit(flush=False)

    def finish(self) -> str:
        """Flush text held back because the last bytes were not a whole character."""
        if not self.token_ids:
            return ""
        return self._emit(flush=True)

    def _emit(self, *, flush: bool) -> str:
        full = self.tokenizer.decode(
            self.token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )
        stable = full if flush else _without_trailing_replacement(full)
        if not stable.startswith(self._emitted):
            raise RuntimeError("detokenized text changed behind the cursor")
        new = stable[len(self._emitted) :]
        self._emitted = stable
        return new


def _without_trailing_replacement(text: str) -> str:
    """Drop U+FFFD only from the end, where an unfinished UTF-8 sequence shows up."""
    end = len(text)
    while end > 0 and text[end - 1] == "\ufffd":
        end -= 1
    return text[:end]
