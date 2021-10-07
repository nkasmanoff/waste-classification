#
# Note -- this training script is tweaked from the original at:
#           https://github.com/pytorch/examples/tree/master/imagenet
#
# For a step-by-step guide to transfer learning with PyTorch, see:
#           https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#
import os
import random
import shutil
import time
import warnings
import wandb # TODO - add loss, val acc.
wandb.login()
import hydra
import logging
<<<<<<< HEAD

logger = logging.getLogger(__name__)
from omegaconf.omegaconf import OmegaConf
=======
from omegaconf.omegaconf import OmegaConf
logger = logging.getLogger(__name__)
>>>>>>> 761d27e52d2c5fd78ad574dcb7b51b08c78e725c


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from reshape import reshape_model

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



best_acc1 = 0


#
# initiate worker threads (if using distributed multi-GPU)
#
@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    #logger.info(f"Using the model: {cfg.model.name}")
    #logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    # instantiate run
#    run = wandb.init(project="waste-classification") # team log

    # add cfg to wandb
    wandb.config.update(cfg, allow_val_change=True)
<<<<<<< HEAD
=======
    run = wandb.init(project="waste-classification") # team log
>>>>>>> 761d27e52d2c5fd78ad574dcb7b51b08c78e725c

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    #if cfg.gpu is not None:
    #    warnings.warn('You have chosen a specific GPU. This will completely '
    #                  'disable data parallelism.')

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if cfg.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.world_size = ngpus_per_node * cfg.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, cfg=(ngpus_per_node, cfg))
    else:
        # Simply call main_worker function
        main_worker(cfg.gpu, ngpus_per_node, cfg)


#
# worker thread (per-GPU)
#
def main_worker(gpu, ngpus_per_node, cfg):
    global best_acc1
    cfg.gpu = gpu

    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)

    # data loading code
    traindir = os.path.join(cfg.data, 'train')
    valdir = os.path.join(cfg.data, 'val')
    testdir = os.path.join(cfg.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # mean and std of imagenet
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            #transforms.Resize(224),
            transforms.RandomResizedCrop(cfg.resolution),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomPerspective(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            normalize,
        ]))

    num_classes = len(train_dataset.classes)
    print('=> dataset classes:  ' + str(num_classes) + ' ' + str(train_dataset.classes))

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(cfg.resolution),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True)


    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(cfg.resolution),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True)


    # create or load the model if using pre-trained (the default)
    if cfg.pretrained:
        print("=> using pre-trained model '{}'".format(cfg.arch))
        model = models.__dict__[cfg.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(cfg.arch))
        model = models.__dict__[cfg.arch]()

    # reshape the model for the number of classes in the dataset
    model = reshape_model(model, cfg.arch, num_classes)

    # transfer the model to the GPU that it should be run on
    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
            cfg.workers = int(cfg.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if cfg.arch.startswith('alexnet') or cfg.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(cfg.gpu)

    optimizer = torch.optim.SGD(model.parameters(), cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

    # optionally resume from a checkpoint
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume)
            cfg.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if cfg.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(cfg.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))

    cudnn.benchmark = True

    # if in evaluation mode, only run validation
    if cfg.evaluate:
        validate(val_loader, model, criterion, num_classes, cfg)
        return


    wandb.watch(model,criterion, log='all',log_freq = 10)

    # train for the specified number of epochs
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)

        # decay the learning rate
        adjust_learning_rate(optimizer, epoch, cfg)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, num_classes, cfg)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, num_classes, cfg)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not cfg.multiprocessing_distributed or (cfg.multiprocessing_distributed
                and cfg.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg.arch,
                'resolution': cfg.resolution,
                'num_classes': num_classes,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, cfg)

    test(test_loader, model, criterion, num_classes, cfg)

#
# train one epoch
#
def train(train_loader, model, criterion, optimizer, epoch, num_classes, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # get the start time
    epoch_start = time.time()
    end = epoch_start

    # train over each image batch from the dataset
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if cfg.gpu is not None:
            images = images.cuda(cfg.gpu, non_blocking=True)
            target = target.cuda(cfg.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, min(5, num_classes)))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # todo - log here
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            progress.display(i)


    wandb.log({'train_loss': float(losses.avg)})
    wandb.log({'train_top1_acc': float(top1.avg)})
    wandb.log({'train_top5_acc':float(top5.avg)})


    print("Epoch: [{:d}] completed, elapsed time {:6.3f} seconds".format(epoch, time.time() - epoch_start))


#
# measure model performance across the val dataset
#
def validate(val_loader, model, criterion, num_classes, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if cfg.gpu is not None:
                images = images.cuda(cfg.gpu, non_blocking=True)
                target = target.cuda(cfg.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, num_classes)))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    wandb.log({"valid loss": float(losses.avg)})
    wandb.log({"valid top1 acc": float(top1.avg)})
    wandb.log({"valid top5 acc": float(top5.avg)})

    return top1.avg


def test(test_loader, model, criterion, num_classes, cfg):

    """
    Following training and validation take best model, and run on test set.
    """

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    if cfg.model_dir:
        model_path = os.path.expanduser(cfg.model_dir)

        best_filename = os.path.join(model_path, 'model_best.pth.tar')
        best_checkpoint = torch.load(best_filename)

    else:
        return "Please make sure model dir is set! Exiting test."

    model.load_state_dict(best_checkpoint['state_dict'])

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            if cfg.gpu is not None:
                images = images.cuda(cfg.gpu, non_blocking=True)
                target = target.cuda(cfg.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, num_classes)))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    wandb.log({"test loss": float(losses.avg)})
    wandb.log({"test top1 acc": float(top1.avg)})
    wandb.log({"test top5 acc": float(top5.avg)})

    return top1.avg



#
# save model checkpoint
#
def save_checkpoint(state, is_best, cfg, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    """Save a model checkpoint file, along with the best-performing model if applicable"""

    # if saving to an output directory, make sure it exists
    if cfg.model_dir:
        model_path = os.path.expanduser(cfg.model_dir)

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        filename = os.path.join(model_path, filename)
        best_filename = os.path.join(model_path, best_filename)

    # save the checkpoint
    torch.save(state, filename)

    # earmark the best checkpoint
    if is_best:
        shutil.copyfile(filename, best_filename)
        print("saved best model to:  " + best_filename)
    else:
        print("saved checkpoint to:  " + filename)


#
# statistic averaging
#
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


#
# progress metering
#
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


#
# learning rate decay
#
def adjust_learning_rate(optimizer, epoch, cfg):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = cfg.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#
# compute the accuracy for a given result
#
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
