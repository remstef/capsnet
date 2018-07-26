# -*- coding: utf-8 -*-

import torch

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Some Capsnet Test')
    parser.add_argument('--in', type=str, default='./data/minisample.txt', required=True, help='location of the input data')
    parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--save', type=str,  default='model.pt', help='path to save the final model')
    args = parser.parse_args()
    
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
      if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
      else:
        torch.cuda.manual_seed(args.seed)
            
    