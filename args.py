import argparse

def make_args():
    parser = argparse.ArgumentParser(description='SR-HAN main.py')
    parser.add_argument('--dataset', type=str, default='CiaoDVD')
    parser.add_argument('--batch', type = int, default=8192, metavar='N', help='input batch size for training')  
    parser.add_argument('--seed', type = int, default=2023, metavar='int', help='random seed')
    parser.add_argument('--decay', type = float, default=0.97, metavar='LR_decay', help='decay')
    parser.add_argument('--lr', type = float, default=0.055, metavar='LR', help='learning rate')
    parser.add_argument('--minlr', type = float,default=0.0001)
    parser.add_argument('--bprreg', type = float, default=0.03)
    parser.add_argument('--projreg', type = float, default=0.01)
    parser.add_argument('--epochs', type = int, default=400, metavar='N', help='number of epochs to train')
    parser.add_argument('--patience', type = int, default=10, metavar='int', help='early stop patience')
    parser.add_argument('--topk', type = int, default=10)
    parser.add_argument('--hide_dim', type = int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--hete_hide_dim', type = int, default=64, metavar='N', help='hete_embedding size')
    parser.add_argument('--Layers', type = int, default=2, help='the numbers of ui-GCN layer') 
    parser.add_argument('--activation', type = str, default='ELU', help='Activation Function of Project layer')   
    parser.add_argument('--FLayers', type = int, default=2, help='the numbers of FeatureGCN layer') 
    args = parser.parse_args()

    return args
