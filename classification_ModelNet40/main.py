"""
Usage:
python main.py --model PointMLP --msg demo
"""
import argparse
import os
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from utils.global_local_loss import ChamferLoss, MetricLoss, NormalLoss
from data import ModelNet40 
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np
from tqdm import tqdm


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='PointNet', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=0.005, type=float, help='min lr')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
    device = 'cuda'
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.model + message + '-' + str(args.seed)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')
    pointmlp = models.__dict__[args.model]()
    criterion = cal_loss
    pointmlp = pointmlp.to(device)
    # criterion = criterion.to(device)
    if device == 'cuda':
        pointmlp = torch.nn.DataParallel(pointmlp)
        cudnn.benchmark = True

    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])
    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        pointmlp.load_state_dict(checkpoint['pointmlp'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']

    printf('==> Preparing data..')
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)

    optimizer = torch.optim.SGD(pointmlp.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(pointmlp, train_loader, optimizer, criterion, device)  # {"loss", "acc", "acc_avg", "time"}
        test_out = validate(pointmlp, test_loader, criterion, device)
        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        save_model(
            pointmlp, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,  # best test accuracy
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"],
                       test_out["loss"], test_out["acc_avg"], test_out["acc"]])
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")
    logger.close()

    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    printf(f"++++++++" * 5)


def train(pointmlp, pointGLR, trainloader, optimizer, criterion, has_normal, device):
    pointmlp.train()
    pointGLR.train()
    metric_criterion = MetricLoss()
    chamfer_criterion = ChamferLoss()

    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in tqdm(enumerate(trainloader)):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 1024]
        optimizer.zero_grad()
        ##############################################
        # We change the output of the model
        logits, features1, fuse_global, normals_pred = pointmlp(data)
        loss_mlp = criterion(logits, label)

        global_feature1 = features1[2].squeeze(2)
        refs1 = features1[0:2]
        recon1 = pointGLR(fuse_global).transpose(1, 2)  # bs, np, 3
        loss_metric = metric_criterion(global_feature1, refs1)
        loss_recon = chamfer_criterion(recon1, points_gt)
        if has_normal:
            loss_normals = NormalLoss(normals_pred, normals)
        else:
            loss_normals = normals_pred.new(1).fill_(0)
        
        loss = loss_recon + loss_metric + loss_normals + loss_mlp
        ##############################################
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pointmlp.parameters(), 1)
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(pointmlp, testloader, criterion, device):
    pointmlp.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in tqdm(enumerate(testloader)):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = pointmlp(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':
    main()
