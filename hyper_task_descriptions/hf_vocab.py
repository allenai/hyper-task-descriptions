from typing import Iterable, Optional, Sequence

import tensorflow.compat.v2 as tf
from seqio import Vocabulary
from transformers import AutoTokenizer


class HuggingfaceVocabulary(Vocabulary):
    """Really simple wrapper around huggingface tokenizer."""

    def __init__(self, model_name: str, extra_ids: int = 0, add_special_tokens: bool = False):
        """Vocabulary constructor.
        Args:
          extra_ids: The number of extra IDs to reserve.
        """
        self._tokenizer = None  # lazy load tokenizer
        self.model_name = model_name
        self._extra_ids = extra_ids or 0
        assert self._extra_ids == 0
        self._add_special_tokens = add_special_tokens
        super().__init__(extra_ids=extra_ids)

    def _load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def unk_id(self) -> Optional[int]:
        return self.tokenizer.unk_token_id

    @property
    def _base_vocab_size(self) -> int:
        """Vocabulary size, excluding extra ids but including PAD/EOS/UNK."""
        return self.tokenizer.vocab_size

    def _encode(self, s: str) -> Sequence[int]:
        return self.tokenizer(s, add_special_tokens=self._add_special_tokens)["input_ids"]

    def _decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def decode(self, ids: Iterable[int]):
        """Detokenizes int32 iterable to a string, up through first EOS."""
        clean_ids = list(ids)

        if self.unk_id is not None:
            vocab_size = self._base_vocab_size
            clean_ids = [self.unk_id if i >= vocab_size else i for i in clean_ids]

        if self.eos_id is not None and self.eos_id in clean_ids:
            clean_ids = clean_ids[: clean_ids.index(self.eos_id) + 1]

        return self._decode(clean_ids)

    def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        def enc(s):
            r = self.tokenizer(
                s.numpy().decode("utf-8"),
                return_tensors="tf",
                add_special_tokens=self._add_special_tokens,
            )["input_ids"]
            return tf.cast(r, tf.int32)

        # we reshape to ensure that we get a 1-dimensional tensor.
        return tf.reshape(tf.py_function(enc, [s], Tout=tf.int32), [-1])

    def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        return tf.constant(self.tokenizer.decode(ids, skip_special_tokens=True))

    def __eq__(self, other):
        # this is an overly simple implementation of __eq__, but should be okay.
        # if not isinstance(other, HuggingfaceVocabulary):
        #     return False
        # try:
        #     their_model_name = other.model_name
        # except AttributeError:
        #     return False
        # return self.model_name == their_model_name
        # hack!
        return True
