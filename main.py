import argparse

from utils.train import train
from utils.test import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='subtitle localtion')
    parser.add_argument('action', help='action type (train/test)')
    parser.add_argument('--resume', action='store_true', help='恢复已有模型继续训练')
    parser.add_argument('--use-cache', action='store_true', help='测试时直接使用已有图片')

    args = parser.parse_args()

    if args.action == "train":
        train(resume=args.resume)
    elif args.action == "test":
        test(use_cache=args.use_cache)
    else:
        print("can not parse arg {}".format(args.action))
