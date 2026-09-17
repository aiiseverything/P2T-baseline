"""Small real tokenizers for canonical chat and exact credit mapping tests."""
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast


class ByteLevelTestTokenizer(PreTrainedTokenizerFast):
    def __init__(self, vocab):
        vocab = {({'\n': 'Ċ', ' ': 'Ġ'}.get(token, token)): index
                 for token, index in vocab.items()}
        backend = Tokenizer(BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
        backend.decoder = ByteLevelDecoder()
        super().__init__(tokenizer_object=backend, eos_token='<|endoftext|>',
                         pad_token='<|endoftext|>',
                         additional_special_tokens=['<|im_end|>'])
        self.chat_template = (
            "{{ messages[0]['content'] }}"
            "{% if messages|length > 1 %}{{ messages[1]['content'] + '<|im_end|>' }}{% endif %}")
