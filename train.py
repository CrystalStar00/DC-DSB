import os
os.environ["MPLBACKEND"] = "Agg"
import torch, argparse, datetime, random, numpy as np
from util.runner import Runner
import sys
import torch.distributed as dist



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_setup():
    torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=2))
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    print(f'Initialized global rank {os.environ["RANK"]}, local rank {os.environ["LOCAL_RANK"]}')


def main(args):

    args.global_rank = int(os.environ.get('RANK', 0))
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.node = int(args.global_rank // args.gpus)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ddp_setup()

    seed_everything(0 + args.global_rank)

    runner = Runner(args)
    # runner.load_ckpt()

    runner.train()

    return


def create_parser():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--gpus', type=int, default=1, help='number of gpus per node')
    argparser.add_argument('--node', type=int, default=0, help='number of nodes')
    argparser.add_argument('--local_rank', type=int, default=0, help='local rank')
    argparser.add_argument('--global_rank', type=int, default=0, help='global rank')

    argparser.add_argument('--method', type=str, default='dsb', help='method')
    argparser.add_argument('--simplify', action='store_true', default=True, help='whether to use simplified DSB')
    argparser.add_argument('--reparam', type=str, default='term',
                           help='whether to use reparameterized DSB, "term" for TR-DSB, "flow" for FR-DSB')
    argparser.add_argument('--noiser', type=str, default='flow',
                           help='noiser type, "flow" noiser for Flow Matching models, "dsb" noiser for DSB models')
    argparser.add_argument('--gamma_type', type=str, default='linear_1e-3_1e-2', help='gamma schedule for DSB')
    argparser.add_argument('--training_timesteps', type=int, default=50, help='training timesteps')
    argparser.add_argument('--inference_timesteps', type=int, default=50, help='inference timesteps')

    argparser.add_argument('--network', type=str, default='gene-transformers', help='network architecture to use')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    argparser.add_argument('--batch_size', type=int, default=2 ** 9, help='batch size')
    argparser.add_argument('--epochs', type=int, default=300, help='number of training epochs')
    argparser.add_argument('--skip_epochs', type=int, default=0, help='number of epochs to skip')
    argparser.add_argument('--repeat_per_epoch', type=float, default=2.0,
                           help='training iteration multiplier per epoch')
    argparser.add_argument('--use_amp', action='store_true', default=True,
                           help='whether to use mixed-precision training')
    # argparser.add_argument('--log_interval', type=int, default=32, help='interval for printing log')

    argparser.add_argument('--data', type=str, default='norman', help='dataset')
    argparser.add_argument('--prior', type=str, default='genepert_train', help='prior distribution')
    argparser.add_argument('--val_prior', type=str, default='genepert_vaild',
                           help='prior distribution for evaluation, only available in image experiments')
    argparser.add_argument('--val_dataset', type=str, default='genepert_vaild',
                           help='data distribution for evaluation, only available in image experiments')

    argparser.add_argument('--wandb', type=bool, default=True, help='whether to use wandb for logging')
    argparser.add_argument('--project', type=str, default='sdsb', help='wandb project name')

    argparser.add_argument('--exp_name', type=str, default='try', help='name of experiment')
    argparser.add_argument('--ckpt', type=str, default=None, help='checkpoint to load')
    argparser.add_argument('--config', type=str, default='./config/model.yaml', help='config file to load')

    return argparser


if __name__ == '__main__':

    argparser = create_parser()
    args = argparser.parse_args()
    if 'dsb' in args.method:
        assert args.training_timesteps == args.inference_timesteps

    main(args)