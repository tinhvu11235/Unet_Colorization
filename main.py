# main.py
import argparse
from train import train_from_scratch, continue_training
from config import TRAIN_FROM_SCRATCH, CONTINUE_TRAINING

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["scratch", "continue"], default="scratch",
                        help="Chọn chế độ huấn luyện: scratch (từ đầu) hoặc continue (tiếp tục từ checkpoint)")
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                        help="Đường dẫn tới file checkpoint nếu tiếp tục huấn luyện (chế độ continue)")
    args = parser.parse_args()

    if args.mode == "scratch":
        print("Huấn luyện từ đầu...")
        train_from_scratch(TRAIN_FROM_SCRATCH)
    elif args.mode == "continue":
        print("Tiếp tục huấn luyện từ checkpoint...")
        continue_training(CONTINUE_TRAINING,args.checkpoint_path)
