import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.val_ar import train
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train binary sentiment classifier.")

    parser.add_argument("--train",               required=True,              help="Path to raw training data.")
    parser.add_argument("--out-path",            required=True,              help="Path to write processed files to.")
    parser.add_argument("--model-path",          required=True,              help="HuggingFace model name or local path.")
    parser.add_argument("--attn-type",   required=True, choices=["self", "cross"])
    parser.add_argument("--attn-layers", required=True, type=int, nargs="+", metavar="LAYER")
    parser.add_argument("--test",                default=None,               help="Path to test file. Triggers evaluation if provided.")
    parser.add_argument("--pretrained-weights",  default=None,               help="Path to saved model weights. Skips training if provided.")
    parser.add_argument("--epochs",              default=3,    type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(
        train=args.train,
        out_path=args.out_path,
        model_path=args.model_path,
        attn_type=args.attn_type,
        attn_layers=args.attn_layers,
        test=args.test,
        pretrained_weights=args.pretrained_weights,
        epochs=args.epochs,
    )