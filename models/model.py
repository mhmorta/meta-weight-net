import torch
from models.wideresnet import WideResNet, VNet
from models.resnet import ResNet32
from torch.autograd import Variable

def build_model(args):
    # model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                       args.widen_factor, dropRate=args.droprate)
    # weights_init(model)

    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.params()])))

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)
