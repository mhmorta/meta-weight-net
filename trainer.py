import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

from models.model import build_model, to_var
from data import build_dataset, build_dataset_continual
from models.wideresnet import VNet
from log import accuracy, AverageMeter

best_prec1 = 0

def train_weighted(args, data_loaders ):
    global best_prec1
    # create model
    train_loaders, train_meta_loader, test_loader = data_loaders 

    model = build_model(args)

    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)


    vnet = VNet(1, 100, 1).cuda()

    optimizer_c = torch.optim.SGD(vnet.params(), 1e-3,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    model_loss = []
    meta_model_loss = []
    smoothing_alpha = 0.9

    meta_l = 0
    net_l = 0
    accuracy_log = []
    train_acc = []

    for train_loader in train_loaders:
        for iters in range(args.iters):
            adjust_learning_rate(args, optimizer_a, iters + 1)
            # adjust_learning_rate(optimizer_c, iters + 1)
            model.train()

            input, target = next(iter(train_loader))
            input_var = to_var(input, requires_grad=False)
            target_var = to_var(target, requires_grad=False)

            meta_model = build_model(args)

            # Why meta_model loads state dict of model?
            meta_model.load_state_dict(model.state_dict())
            y_f_hat = meta_model(input_var)
            cost = F.cross_entropy(y_f_hat, target_var, reduce=False)
            cost_v = torch.reshape(cost, (len(cost), 1))


            v_lambda = vnet(cost_v.data)

            norm_c = torch.sum(v_lambda)

            if norm_c != 0:
                v_lambda_norm = v_lambda / norm_c
            else:
                v_lambda_norm = v_lambda

            l_f_meta = torch.sum(cost_v * v_lambda_norm)
            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta,(meta_model.params()),create_graph=True)
            meta_lr = args.lr * ((0.1 ** int(iters >= 18000)) * (0.1 ** int(iters >= 19000)))  # For WRN-28-10
            #meta_lr = args.lr * ((0.1 ** int(iters >= 20000)) * (0.1 ** int(iters >= 25000)))  # For ResNet32
            meta_model.update_params(lr_inner=meta_lr,source_params=grads)
            del grads

            input_validation, target_validation = next(iter(train_meta_loader))
            input_validation_var = to_var(input_validation, requires_grad=False)
            target_validation_var = to_var(target_validation.type(torch.LongTensor), requires_grad=False)

            y_g_hat = meta_model(input_validation_var)
            l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
            prec_meta = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]


            optimizer_c.zero_grad()
            l_g_meta.backward()
            optimizer_c.step()


            y_f = model(input_var)
            cost_w = F.cross_entropy(y_f, target_var, reduce=False)
            cost_v = torch.reshape(cost_w, (len(cost_w), 1))
            prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]


            with torch.no_grad():
                w_new = vnet(cost_v)
            norm_v = torch.sum(w_new)

            if norm_v != 0:
                w_v = w_new / norm_v
            else:
                w_v = w_new

            l_f = torch.sum(cost_v * w_v)


            optimizer_a.zero_grad()
            l_f.backward()
            optimizer_a.step()

            meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
            meta_model_loss.append(meta_l / (1 - smoothing_alpha ** (iters + 1)))

            net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
            model_loss.append(net_l / (1 - smoothing_alpha ** (iters + 1)))


            if (iters + 1) % 100 == 0:
                print('Epoch: [%d/%d]\t'
                    'Iters: [%d/%d]\t'
                    'Loss: %.4f\t'
                    'MetaLoss:%.4f\t'
                    'Prec@1 %.2f\t'
                    'Prec_meta@1 %.2f' % (
                        (iters + 1) // 500 + 1, args.epochs, iters + 1, args.iters, model_loss[iters],
                        meta_model_loss[iters], prec_train, prec_meta))

                losses_test = AverageMeter()
                top1_test = AverageMeter()
                model.eval()


                for i, (input_test, target_test) in enumerate(test_loader):
                    input_test_var = to_var(input_test, requires_grad=False)
                    target_test_var = to_var(target_test, requires_grad=False)

                    # compute output
                    with torch.no_grad():
                        output_test = model(input_test_var)
                    loss_test = criterion(output_test, target_test_var)
                    prec_test = accuracy(output_test.data, target_test_var.data, topk=(1,))[0]

                    losses_test.update(loss_test.data.item(), input_test_var.size(0))
                    top1_test.update(prec_test.item(), input_test_var.size(0))

                print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1_test))

                accuracy_log.append(np.array([iters, top1_test.avg])[None])
                train_acc.append(np.array([iters, prec_train])[None])

                best_prec1 = max(top1_test.avg, best_prec1)
    return meta_model_loss, model_loss, accuracy_log, train_acc
        

def adjust_learning_rate(args, optimizer, iters):

    lr = args.lr * ((0.1 ** int(iters >= 18000)) * (0.1 ** int(iters >= 19000)))  # For WRN-28-10
    #lr = args.lr * ((0.1 ** int(iters >= 20000)) * (0.1 ** int(iters >= 25000)))  # For ResNet32
    # log to TensorBoard
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
