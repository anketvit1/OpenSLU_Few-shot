'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-02-19 19:01:40
Description: main executive file

'''
import argparse

from common.config import Config
from common.model_manager import ModelManager
import os

# Set CUDA_VISIBLE_DEVICES to select a specific GPU (e.g., GPU 2)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-cp', type=str, default='config/stack-propagation.yaml')
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--dataset', '-ds', type=str, default=None)
    parser.add_argument('--device', '-dv', type=str, default=None)
    parser.add_argument('--learning_rate', '-lr', type=float, default=None)
    parser.add_argument('--epoch_num', '-en', type=int, default=None)
    parser.add_argument('--data_fraction', '-df', type=float, default=None)
    args = parser.parse_args()
    config = Config.load_from_args(args)
    model_manager = ModelManager(config, data_fraction=args.data_fraction)
    model_manager.init_model()
    if config.base.get("train"):
        model_manager.train()
    if not config.base.get("train") and config.base.get("test"):
        model_manager.test()


if __name__ == "__main__":
    main()
