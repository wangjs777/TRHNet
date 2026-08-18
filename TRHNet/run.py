#!/aaw/anaconda/envs/CR7/bin/python3.8
# import msadapter.pytorch as torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
# import mindspore as ms
import argparse
from loguru import logger
from data.data_loader import load_data as load_data
import os
import numpy as np
import random
import A_2_Net_train_plus as A_2_Net_plus

import warnings

warnings.filterwarnings("ignore")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run():
    # seed_everything(68)
    seed_everything(3)
    args = load_config()
    if args.arch != 'test':
        if not os.path.exists('logs/' + '2'):
            os.makedirs('logs/' + '2')
        # logger.add('logs/' + args.info + '/-{time}_' + args.arch + '_' + args.dataset + '.log', rotation='500 MB', level='INFO')
        # logger.add('logs/' + 2.info + '/' + str(args.code_length) + '/food101-finet.log')
        logger.add('logs/' + '2' + '/' + str(args.code_length) + '/' + 'food101_2.log')
        # logger.add('logs/'+args.info+'/'+str(args.code_length)+'/resonly-test.log')
        logger.info(args)

    torch.backends.cudnn.benchmark = True

    # Load dataset
    query_dataloader, train_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        # args.num_query,
        # args.num_samples,
        args.batch_size,
        args.num_workers,
    )

    if args.arch == 'a2net':
        net_arch = A_2_Net_plus
    elif args.arch == 'a2net_ft':
        net_arch = A_2_Net_finetuneO

    # for code_length in args.code_length:
    #     mAP = net_arch.train_e2e(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args)
    #     logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, mAP))
    for code_length in args.code_length:

        # =====================================================
        # Complexity-only mode
        # =====================================================
        if args.complexity_only:

            if args.checkpoint == '':
                checkpoint_path = os.path.join(
                    'checkpoints_test',
                    args.info,
                    str(code_length),
                    'model.pkl'
                )
            else:
                checkpoint_path = args.checkpoint

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}"
                )

            complexity = net_arch.test_complexity(
                query_dataloader=query_dataloader,
                code_length=code_length,
                args=args,
                checkpoint_path=checkpoint_path
            )

            logger.info(
                f"[FINAL]"
                f"[code_length:{code_length}]"
                f"[Params:{complexity['params_m']:.3f} M]"
                f"[FLOPs:{complexity['flops_g']:.3f} G]"
                f"[Latency:{complexity['latency_ms']:.4f} ms/image]"
            )

        # =====================================================
        # Normal training
        # =====================================================
        else:

            mAP = net_arch.train_e2e(
                query_dataloader,
                train_dataloader,
                retrieval_dataloader,
                code_length,
                args
            )

            logger.info(
                '[code_length:{}][map:{:.4f}]'.format(
                    code_length,
                    mAP
                )
            )


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='ADSH_PyTorch')
    parser.add_argument('--dataset', default="food-101", type=str,
                        help='Dataset name.')
    parser.add_argument('--root', default="/22wangjunsong/datasets/food-101/",
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size.(default: 16)')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='Weight Decay.(default: 1e-4)')
    parser.add_argument('--optim', default='SGD', type=str,
                        help='Optimizer')
    parser.add_argument('--code-length', default='12', type=str,
                        help='Binary hash code length.(default: 12,24,32,48)')
    parser.add_argument('--max-epoch', default=150, type=int,
                        help='Number of epochs.(default: 5)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='Number of loading data threads.(default: 4)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--gamma', default=1, type=float,
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--info', default='Train',
                        help='Train info')
    parser.add_argument('--arch', default='a2net',
                        help='Net arch')
    parser.add_argument('--save_ckpt', default='checkpoints/',
                        help='result_save')
    parser.add_argument('--lr-step', default='120', type=str,
                        help='lr decrease step.(default: 40,50)')
    parser.add_argument('--align-step', default=50, type=int,
                        help='Step of start aligning.(default: 50)')
    parser.add_argument('--pretrain', default=True
                        , action='store_true',
                        help='Using image net pretrain')
    parser.add_argument('--lambd-sp', default=1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--lambd-ch', default=1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--lambd', default=1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--momen', default=0.91, type=float,
                        help='Hyper-parameter.(default: 0.9)')
    parser.add_argument('--nesterov', default=True, action='store_true',
                        help='Using SGD nesterov')
    parser.add_argument('--num_classes', default=101, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--margin', default=4, type=float)
    parser.add_argument('--eval_freq', default=30, type=int)
    parser.add_argument(
        '--complexity-only',
        action='store_true',
        help='Only evaluate inference complexity without training.')

    parser.add_argument(
        '--checkpoint',
        default='/model.pkl',
        type=str,
        help='Path to trained model checkpoint.')
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
    # args.device = torch.device("cuda:0")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))
    args.lr_step = list(map(int, args.lr_step.split(',')))

    return args


if __name__ == '__main__':
    run()
