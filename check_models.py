"""Check model tokenizer compatibility using public/local metadata only."""
import argparse
from transformers import AutoTokenizer, PretrainedConfig
from vpo_rm import check_tokenizers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor', required=True)
    parser.add_argument('--rm', required=True)
    parser.add_argument('--pad-token', help='Register the same existing/reserved padding token')
    args = parser.parse_args()
    actor_tok = AutoTokenizer.from_pretrained(args.actor)
    rm_tok = AutoTokenizer.from_pretrained(args.rm)
    if args.pad_token:
        for tok in (actor_tok, rm_tok):
            tok.add_special_tokens({'pad_token': args.pad_token})
    ac, _ = PretrainedConfig.get_config_dict(args.actor)
    rc, _ = PretrainedConfig.get_config_dict(args.rm)
    check_tokenizers(actor_tok, rm_tok, ac['vocab_size'], rc['vocab_size'])
    print(f"Tokenizers aligned; model vocabulary rows: {ac['vocab_size']}; "
          f"padding token ID: {actor_tok.pad_token_id}")


if __name__ == '__main__':
    main()
