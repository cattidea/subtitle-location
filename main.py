import argparse

from utils.train import train
from utils.test import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='subtitle localtion')
    parser.add_argument('action', help='action type (train/test)')
    parser.add_argument('--resume', action='store_true', help='恢复已有模型继续训练')
    parser.add_argument('--use-cache', action='store_true', help='测试时直接使用已有图片(暂时取消)')
    parser.add_argument('--no-rec', action='store_true', default=False, help='取消删除')
    parser.add_argument('--no-gpu', action='store_true', default=False, help='不使用 GPU')

    args = parser.parse_args()
    train_config = {
        "resume": args.resume,
        "use_GPU": not args.no_gpu,
    }
    test_config = {
        "use_cache": args.use_cache,
        "recognition": not args.no_rec,
        "use_GPU": not args.no_gpu,
    }

    if args.action == "train":
        train(train_config)
    elif args.action == "test":
        test(test_config)
    else:
        print("can not parse arg {}".format(args.action))
