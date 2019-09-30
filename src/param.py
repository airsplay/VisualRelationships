import argparse
import torch
import os

        
def parse_args():
    parser = argparse.ArgumentParser()

    # Principal
    parser.add_argument('--dataset', type=str, default='nlvr')
    parser.add_argument('--train', type=str, default='speaker')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--output', type=str, default='/tmp')
    parser.add_argument('--model', type=str, default='init')

    # MISC
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument("--fast", action='store_const', default=False, const=True)

    # Preprocessing
    parser.add_argument('--imgType', dest='img_type', type=str, default='img')   # img, pixel, feat
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--maxInput', dest='max_input', type=int, default=40)

    # Training Parameters
    parser.add_argument('--seed', type=int, default=9595, help='random seed')
    parser.add_argument('--epochs', type=int, default=950)
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=128)
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)

    # Model Parameters
    parser.add_argument('--hidDim', dest='hid_dim', type=int, default=512)
    parser.add_argument('--embDim', dest='emb_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--featdropout', type=float, default=0.0)
    parser.add_argument("--dropbatch", action='store_const', default=False, const=True)
    parser.add_argument("--dropinstance", action='store_const', default=False, const=True)

    # Speaker configuration
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="one")
    parser.add_argument("--baseline", type=str, default="none")
    parser.add_argument("--metric", type=str, default="ROUGE_L")
    parser.add_argument("--pretrain", action='store_const', default=False, const=True)

    parser.add_argument("--trueSpeaker", dest='true_speaker', type=str, default=None)

    args = parser.parse_args()

    if args.optim == 'rms':
        print("Optimizer: Using RMSProp")
        args.optimizer = torch.optim.RMSprop
    elif args.optim == 'adam':
        print("Optimizer: Using Adam")
        args.optimizer = torch.optim.Adam
    elif args.optim == 'adamax':
        print("Optimizer: Using Adamax")
        args.optimizer = torch.optim.Adamax
    elif args.optim == 'sgd':
        print("Optimizer: sgd")
        args.optimizer = torch.optim.SGD
    else:
        assert False
    return args

args = parse_args()
