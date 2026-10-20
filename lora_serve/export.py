"""Merge the LoRA adapter into the base weights and save a standalone model for serving."""

import argparse

from .modeling import BASE_MODEL, load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapter", default="artifacts/adapter")
    ap.add_argument("--out", default="artifacts/merged")
    args = ap.parse_args(argv)
    from peft import PeftModel

    tok, base = load(args.model)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
