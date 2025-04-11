import argparse
from train import train_from_scratch, train_from_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["scratch", "continue"], default="scratch",
                        help="Select training mode: scratch (from the beginning) or continue (resume from checkpoint)")
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                        help="Path to the checkpoint file if continuing training (mode 'continue')")
    args = parser.parse_args()

    if args.mode == "scratch":
        print("Training from scratch...")
        train_from_scratch()
    elif args.mode == "continue":
        print("Continuing training from checkpoint...")
        train_from_checkpoint(args.checkpoint_path)
