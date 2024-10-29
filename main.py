import argparse
import os

from model.MyGAN import MyGAN
from utils import *
from utils.funs import check_folder


def parse_args():
    desc='pytorch of My-CartoonGAN'
    parser=argparse.ArgumentParser(desc)
    parser.add_argument('--device',type=str,default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--input_c', type=int, default=3)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--pre_epoch', type=int, default=20)
    parser.add_argument('--cpu_count', type=int, default=10)
    parser.add_argument('--init_lr',type=float,default=0.002)
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for ADAM')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for ADAM')
    parser.add_argument('--decay_lr', type=float, default=0.0002, help='learning rate for ADAM')#3.000000000000006e-06
    parser.add_argument('--hw', type=int, default=256)
    parser.add_argument('--result_dir',type=str,default='results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Name of checkpoint directory')
    parser.add_argument('--dataset', type=str, default='hayao')
    parser.add_argument('--data_dir',type=str,default='data')
    parser.add_argument('--test_dir',type=str,default='checkpoint_hayao80.pth')
    parser.add_argument('--isTrain',type=bool,default=False)
    parser.add_argument('--retrain', type=bool, default=True)
    parser.add_argument('--isTest', type=bool, default=True)
    parser.add_argument('--train_init', type=bool, default=False)
    parser.add_argument('--b1',type=int,default=0.5)
    parser.add_argument('--b2', type=int, default=0.999)
    parser.add_argument('--y1', type=int, default=1)
    parser.add_argument('--y2', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--s', type=int, default=48)
    parser.add_argument('--batch_size',type=int,default=6)
    parser.add_argument('--save_pred',type=int,default=1)
    parser.add_argument('--weight_content',type=float,default=2)#big
    parser.add_argument('--weigh'
                        't_struct',type=float,default=2)
    parser.add_argument('--weight_surface', type=float, default=5)
    parser.add_argument('--weight_testure', type=float, default=5)
    parser.add_argument('--weight_classifer', type=float, default=0)
    parser.add_argument('--weight_tv', type=float, default=1)
    parser.add_argument('--weight_style', type=float, default=2)
    parser.add_argument('--weight_edge', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    return check_args(parser.parse_args(args=[]))
def check_args(args):
    check_folder(os.path.join(args.result_dir, args.dataset, 'checkpoint'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'con'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'sty'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'atten'))
    return args
from Module.utm import utm

def main():

   args=parse_args()
   args.isTrain=True
   args.train_init=False
   args.retrain=False
   args.neck = utm
   args.dataset="hayao"
   gan=MyGAN(args)
   if args.isTrain:
       if args.retrain:
            gan.load_model()
       print(f"training on {args.device}")
       gan.train()
       print("train haved finished")
   if args.isTest:
       gan.test()
       print("test haved finished")
if __name__=="__main__":
    main()
