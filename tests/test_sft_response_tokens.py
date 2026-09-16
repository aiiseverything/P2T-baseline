import unittest
from pathlib import Path

from transformers import AutoTokenizer

from scripts.sft_response_tokens import (
    build_response_eos_spec,
    response_labels,
    rewrite_response_eos,
    supervised_token_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


class ResponseEosTokensTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "models/Qwen3-14B-Base", local_files_only=True
        )
        cls.im_end = cls.tokenizer.convert_tokens_to_ids("<|im_end|>")
        cls.native_eos = cls.tokenizer.eos_token_id
        cls.newline = cls.tokenizer("\n", add_special_tokens=False)["input_ids"]

    def example_ids(self):
        prompt = [101, self.im_end, 102]
        content = [201, self.im_end, 202]
        return prompt, prompt + content + [self.im_end] + self.newline

    def test_chat_template_mode_keeps_existing_tokens(self):
        prompt, full = self.example_ids()
        spec = build_response_eos_spec(self.tokenizer, "chat_template")
        self.assertEqual(rewrite_response_eos(full, len(prompt), spec), full)
        self.assertEqual(spec.actual_eos_id, self.im_end)

    def test_native_mode_replaces_only_final_response_ending(self):
        prompt, full = self.example_ids()
        spec = build_response_eos_spec(self.tokenizer, "native")
        actual = rewrite_response_eos(full, len(prompt), spec)
        self.assertEqual(actual[:-len(self.newline) - 1], full[:-len(self.newline) - 1])
        self.assertEqual(actual[-len(self.newline) - 1], self.native_eos)
        self.assertEqual(actual[-len(self.newline):], self.newline)
        self.assertEqual(actual[:len(prompt)], prompt)
        self.assertEqual(actual[len(prompt) + 1], self.im_end)

    def test_native_eos_equal_to_pad_is_still_a_supervised_label(self):
        prompt, full = self.example_ids()
        self.assertEqual(self.native_eos, self.tokenizer.pad_token_id)
        spec = build_response_eos_spec(self.tokenizer, "native")
        actual = rewrite_response_eos(full, len(prompt), spec)
        labels = response_labels(actual, len(prompt))
        eos_position = len(actual) - len(self.newline) - 1
        self.assertEqual(labels[eos_position], self.native_eos)
        self.assertNotEqual(labels[eos_position], -100)

    def test_supervised_token_hash_records_response_eos_choice(self):
        prompt, full = self.example_ids()
        chat = rewrite_response_eos(
            full, len(prompt), build_response_eos_spec(self.tokenizer, "chat_template")
        )
        native = rewrite_response_eos(
            full, len(prompt), build_response_eos_spec(self.tokenizer, "native")
        )
        chat_labels = response_labels(chat, len(prompt))
        native_labels = response_labels(native, len(prompt))
        self.assertNotEqual(
            supervised_token_sha256([chat_labels]),
            supervised_token_sha256([native_labels]),
        )
        self.assertEqual(
            supervised_token_sha256([native_labels]),
            supervised_token_sha256([native_labels]),
        )

    def test_native_mode_rejects_invalid_template_tail(self):
        prompt, full = self.example_ids()
        spec = build_response_eos_spec(self.tokenizer, "native")
        with self.assertRaisesRegex(ValueError, "template tail"):
            rewrite_response_eos(full[:-1], len(prompt), spec)

    def test_native_mode_rejects_missing_native_eos(self):
        tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "models/Qwen3-14B-Base", local_files_only=True
        )
        tokenizer.eos_token = None
        with self.assertRaisesRegex(ValueError, "native EOS"):
            build_response_eos_spec(tokenizer, "native")


if __name__ == "__main__":
    unittest.main()
