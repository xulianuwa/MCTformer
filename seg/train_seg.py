from torch.backends import cudnn

cudnn.enabled = True
from tool import pyutils, torchutils
import argparse
import importlib
import tool.exutils as exutils

import torch.nn.functional as F
from pathlib import Path

import torch
import os
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU_id')

    parser.add_argument("--list_path", default="voc12/train_aug_id.txt", type=str)
    parser.add_argument("--img_path", default="", type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--seg_pgt_path", default=None, type=str)

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--num_epochs", default=15, type=int)
    parser.add_argument("--network", default='', type=str)
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--wt_dec", default=1e-5, type=float)
    parser.add_argument("--init_weights", default='', type=str)

    parser.add_argument("--session_name", default="model_", type=str)
    parser.add_argument("--crop_size", default=321, type=int)

    parser.add_argument('--print_intervals', type=int, default=50)

    args = parser.parse_args()

    gpu_id = args.gpu_ids
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    save_path = args.save_path
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    pyutils.Logger(os.path.join(args.save_path, args.session_name + '.log'))

    criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='elementwise_mean').cuda()

    model = getattr(importlib.import_module('network.' + args.network), 'Net')(num_classes=args.num_classes)

    weights_dict = torch.load(args.init_weights)
    model.load_state_dict(weights_dict, strict=False)

    img_list = exutils.read_file(args.list_path)
    train_size = len(img_list)
    num_batches_per_epoch = train_size // args.batch_size
    max_step = args.num_epochs * num_batches_per_epoch

    data_list = []
    for i in range(200):
        np.random.shuffle(img_list)
        data_list.extend(img_list)

    optimizer = torchutils.PolyOptimizer_cls([
        {'params': model.get_1x_lr_params(), 'lr': args.lr},
        {'params': model.get_10x_lr_params(), 'lr': 10 * args.lr}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    data_gen = exutils.chunker(data_list, args.batch_size)

    for ep in range(args.num_epochs):
        for iter in range(num_batches_per_epoch):
            chunk = data_gen.__next__()
            img_list = chunk

            images, ori_images, seg_labels, img_names = exutils.get_data_from_chunk(chunk, args)

            b, _, w, h = ori_images.shape
            seg_labels = seg_labels.long().cuda()
            images = images.cuda()

            pred = model(x=images)
            pred = F.interpolate(pred, size=(w, h), mode='bilinear', align_corners=False)
            loss = criterion(pred, seg_labels)

            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % args.print_intervals == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.5f' % (optimizer.param_groups[0]['lr']), flush=True)

        torch.save(model.module.state_dict(), os.path.join(save_path, args.session_name + str(ep) + '.pth'))