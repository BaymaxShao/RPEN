import argparse


class Trainer:
    def __init__(self):
        self.configs = None
        self.parser = argparse.ArgumentParser(
            description='NEPose Trainer',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('data', help='path to dataset')
        self.parser.add_argument('--model', help='model to use')
        self.parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                                 help='number of data loading workers')
        self.parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR',
                                 help='initial learning rate')
        self.parser.add_argument('--num_heads', default=12, type=int, help='number of heads in multi-head attention module')
        self.parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH',
                                 help='path to pre-trained Pose net model')
        self.parser.add_argument('--loss', default='QuatLoss', type=str, help='loss function used to train the model')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                                 help='momentum for sgd, alpha parameter for adam')
        self.parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
        self.parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
        self.parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
        self.parser.add_argument('--save_path', type=str, help='path to save models')

    def parse(self, *args, **kwargs):
        self.configs = self.parser.parse_args(*args, **kwargs)
        return self.configs


class Tester:
    def __init__(self):
        self.configs = None
        self.parser = argparse.ArgumentParser(
            description='NEPose Tester',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('data', help='path to dataset')
        self.parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                                 help='number of data loading workers')
        self.parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')
        self.parser.add_argument('--num_heads', default=12, type=int, help='number of heads in multi-head attention module')
        self.parser.add_argument('--pretrained-model1', type=str, help='path to pre-trained model')
        self.parser.add_argument('--pretrained-model2', type=str, help='path to pre-trained model')
        self.parser.add_argument('--save_path', type=str, help='path to save results')

    def parse(self, *args, **kwargs):
        self.configs = self.parser.parse_args(*args, **kwargs)
        return self.configs

