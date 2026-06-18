import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bin import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train binary sentiment classifier.")

    parser.add_argument("--train",               required=True,              help="Path to raw training data.")
    parser.add_argument("--out-path",            required=True,              help="Path to write processed files to.")
    parser.add_argument("--model-path",          required=True,              help="HuggingFace model name or local path.")
    parser.add_argument("--test",                default=None,               help="Path to test file. Triggers evaluation if provided.")
    parser.add_argument("--pretrained-weights",  default=None,               help="Path to saved model weights. Skips training if provided.")
    parser.add_argument("--epochs",              default=3,    type=int)
    parser.add_argument("--domain",              default="",    type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(
        train=args.train,
        out_path=args.out_path,
        model_path=args.model_path,
        test=args.test,
        pretrained_weights=args.pretrained_weights,
        epochs=args.epochs,
        domain=args.domain,
    )