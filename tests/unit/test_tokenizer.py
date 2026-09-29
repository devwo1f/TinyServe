"""Tokenizer round-trips and streaming detokenization.

The byte-level fixture is trained in the test so CI can check multi-byte
behavior without the gated Llama tokenizer. The Llama tests run wherever
`scripts/download_models.sh` has been used.
"""

from pathlib import Path

import pytest

from tinyserve.model.tokenizer import IncrementalDetokenizer, Tokenizer

REPO = Path(__file__).resolve().parents[2]
LLAMA_DIR = REPO / "models" / "Llama-3.2-1B-Instruct"
needs_llama = pytest.mark.skipif(
    not (LLAMA_DIR / "tokenizer.json").exists(),
    reason="Llama tokenizer not downloaded (bash scripts/download_models.sh)",
)

TEXTS = ["Hello, world!", "café", "你好", "a  b", "🌟", "line\nbreak", "naïve — 42"]


def _byte_level_tokenizer() -> Tokenizer:
    """A tiny byte-level BPE, same family as Llama, with no network and no gated files."""
    from tokenizers import Tokenizer as RawTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    raw = RawTokenizer(BPE(unk_token=None))
    raw.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw.decoder = ByteLevelDecoder()
    raw.train_from_iterator(
        ["hello world", "café 你好 🌟", "the quick brown fox"],
        trainer=BpeTrainer(
            vocab_size=400,
            special_tokens=["<s>", "</s>"],
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    backend = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="<s>",
        eos_token="</s>",
        clean_up_tokenization_spaces=False,
    )
    backend.chat_template = (
        "{%- for m in messages -%}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    return Tokenizer(backend)


@pytest.fixture(scope="module")
def byte_tokenizer() -> Tokenizer:
    return _byte_level_tokenizer()


def test_round_trip(byte_tokenizer: Tokenizer):
    for text in TEXTS:
        ids = byte_tokenizer.encode(text)
        assert byte_tokenizer.decode(ids) == text


def test_incremental_matches_full_decode_and_hides_replacement_char(byte_tokenizer: Tokenizer):
    text = "café 你好 🌟"
    ids = byte_tokenizer.encode(text)
    detok = IncrementalDetokenizer(byte_tokenizer)
    pieces = [detok.add(i) for i in ids]
    assert "".join(pieces) + detok.finish() == text
    assert all("\ufffd" not in piece for piece in pieces)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / Llama byte-to-character map. Alphabet order is not byte order."""
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    bytes_ = visible[:]
    chars = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in bytes_:
            bytes_.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(bytes_, [chr(c) for c in chars], strict=True))


def test_partial_utf8_is_held_until_the_character_completes(byte_tokenizer: Tokenizer):
    """Feed the three UTF-8 bytes of 你 as separate tokens, the way a split character arrives."""
    mapping = _bytes_to_unicode()
    byte_ids = [byte_tokenizer.backend.convert_tokens_to_ids(mapping[b]) for b in "你".encode()]
    assert all(isinstance(i, int) and i >= 0 for i in byte_ids)

    detok = IncrementalDetokenizer(byte_tokenizer)
    pieces = [detok.add(i) for i in byte_ids]
    assert pieces[0] == ""
    assert "".join(pieces) == "你"
    assert detok.finish() == ""


def test_chat_template_inserts_roles_and_generation_prompt(byte_tokenizer: Tokenizer):
    ids = byte_tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hi"}],
        add_generation_prompt=True,
    )
    text = byte_tokenizer.decode(ids)
    assert "<|user|>Say hi<|end|>" in text
    assert text.endswith("<|assistant|>")


@needs_llama
def test_llama_round_trip_and_chat_template():
    tok = Tokenizer.from_pretrained(str(LLAMA_DIR))
    for text in TEXTS:
        assert tok.decode(tok.encode(text)) == text

    ids = tok.apply_chat_template(
        [{"role": "user", "content": "Say hi"}],
        add_generation_prompt=True,
    )
    text = tok.decode(ids)
    assert text.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>user<|end_header_id|>\n\nSay hi<|eot_id|>" in text
    assert "<|start_header_id|>assistant<|end_header_id|>" in text
    assert tok.bos_token_id == ids[0]
    assert tok.eos_token_id == tok.encode("<|eot_id|>", add_special_tokens=False)[0]


@needs_llama
def test_llama_incremental_non_ascii():
    tok = Tokenizer.from_pretrained(str(LLAMA_DIR))
    text = "café 你好 🌟 — naïve"
    ids = tok.encode(text)
    detok = IncrementalDetokenizer(tok)
    pieces = [detok.add(i) for i in ids]
    assert "".join(pieces) + detok.finish() == text
    assert all("\ufffd" not in piece for piece in pieces)
    # The emoji is more than one token, so at least one step must hold text back.
    assert "" in pieces
