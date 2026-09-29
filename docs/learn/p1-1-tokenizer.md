# P1.1 Tokenizer and streaming detokenization

## 1. Concept in plain words

The model never sees text. It sees integers. The tokenizer turns text into those integers and back.

Llama 3 uses byte-level BPE. Every UTF-8 byte can be its own token, and common byte sequences get merged into bigger tokens. That means one character is not always one token. "🌟" is three tokens in the Llama 3.2 tokenizer. After the first of those three, the bytes so far are not a finished character. Decoding them produces the replacement character U+FFFD (the box you see for bad text). The next token replaces that box with the real emoji. If a streaming server sent the box, the user would see it flash and disappear.

`IncrementalDetokenizer` decodes the whole sequence on every new token, drops a trailing U+FFFD, and sends only the part that was not sent before. `finish()` flushes whatever is left when generation stops.

The chat template is separate from encode. Llama 3 Instruct wraps each turn in `<|start_header_id|>role<|end_header_id|>` and ends a prompt on the assistant header so the model writes the reply. `encode` does not add the beginning-of-sequence token, because the template already does; adding it twice would shift every position.

## 2. Where it lives in the code

1. `tinyserve/model/tokenizer.py`: `Tokenizer.from_pretrained`, `encode`, `decode`, `apply_chat_template`.
2. Same file: `IncrementalDetokenizer.add` and `finish`.
3. `tests/unit/test_tokenizer.py`: a tiny byte-level tokenizer for CI, plus Llama tests that run when the 1B model has been downloaded.

## 3. Key tensors and shapes

No tensors yet. `encode` and `apply_chat_template` return `list[int]`, one id per token. The first chat-template id for Llama 3 is `<|begin_of_text|>` (128000).

## 4. What was measured

Nothing. No benchmark in this task.

## 5. Pitfalls hit

- `transformers` 5 returns a `BatchEncoding` from `apply_chat_template`, not a plain list. The wrapper reads `input_ids`.
- `clean_up_tokenization_spaces=True` (the checkpoint default) is a WordPiece cleanup. It would delete spaces before punctuation in BPE text, so it is forced off.
- `ByteLevel.alphabet()[i]` is not the character for byte `i`. The byte-to-character map used by GPT-2 and Llama skips control bytes. The test uses that map; indexing the alphabet by the raw byte decoded the wrong token.

## 6. Self-check questions

1. Why can one Unicode character be several tokens?
2. Why must streaming detokenization hold back U+FFFD instead of sending it?
3. Why does `encode` not add the BOS token by default?
4. What does `add_generation_prompt=True` put at the end of a Llama 3 chat prompt, and why?
5. Why is the Llama tokenizer test skipped in CI?

<details>
<summary>Answers</summary>

1. Byte-level BPE starts from UTF-8 bytes and merges common sequences. A rare character, or an emoji, may never have been merged into one token, so it stays as several byte tokens.
2. U+FFFD is the decoder's stand-in for an unfinished byte sequence. The next token often completes the character and replaces it. Sending it would show a box that then vanishes.
3. The chat template already inserts `<|begin_of_text|>`. Adding it again in `encode` would duplicate it and shift every later position.
4. It ends on `<|start_header_id|>assistant<|end_header_id|>` so the model continues with the assistant reply instead of seeing a closed user turn.
5. The tokenizer files are part of a gated Llama download and are gitignored. CI has no `HF_TOKEN` and no weights. The same streaming behavior is tested there with a tiny byte-level tokenizer built in the test.

</details>
