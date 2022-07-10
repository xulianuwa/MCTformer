import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import importlib
from tool import imutils
import argparse
import cv2
import os.path
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from tool.metrics import Evaluator
import PIL.Image as Image

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

classes = np.array(('background',  # always index 0
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor'))


def crf_postprocess(pred_prob, ori_img):
    crf_score = imutils.crf_inference_inf(ori_img, pred_prob, labels=21)
    return crf_score

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1','True'):
        return True
    elif v.lower() in ('no','false','f','n','0','False'):
        return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU_id')
    parser.add_argument("--weights", default="", type=str)
    parser.add_argument("--network", default="", type=str)
    parser.add_argument("--gt_path", required=True, type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--save_path_c", default=None, type=str)
    parser.add_argument("--list_path", default="./voc12/val_id.txt", type=str)
    parser.add_argument("--img_path", default="", type=str)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--use_crf", default=False, type=str2bool)
    parser.add_argument("--scales", type=float, nargs='+')
    args = parser.parse_args()

    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    Path(args.save_path_c).mkdir(parents=True, exist_ok=True)

    model = getattr(importlib.import_module('network.' + args.network), 'Net')(num_classes=args.num_classes)

    model.load_state_dict(torch.load(args.weights))
    seg_evaluator = Evaluator(num_class=args.num_classes)
    model.eval()
    model.cuda()
    im_path = args.img_path
    img_list = open(args.list_path).readlines()

    with torch.no_grad():
        for idx in tqdm(range(len(img_list))):
            i = img_list[idx]

            img_temp = cv2.imread(os.path.join(im_path, i.strip() + '.jpg'))
            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float)
            img_original = img_temp.astype(np.uint8)

            img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
            img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
            img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

            input = torch.from_numpy(img_temp[np.newaxis, :].transpose(0, 3, 1, 2)).float().cuda()

            N, C, H, W = input.size()

            probs = torch.zeros((N, args.num_classes, H, W)).cuda()
            if args.scales:
                scales = tuple(args.scales)

            for s in scales:
                new_hw = [int(H * s), int(W * s)]
                im = F.interpolate(input, new_hw, mode='bilinear', align_corners=True)
                prob = model(x=im)

                prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=False)
                prob = F.softmax(prob, dim=1)
                probs = torch.max(probs, prob)

            output = probs.cpu().data[0].numpy()

            if args.use_crf:
                crf_output = crf_postprocess(output, img_original)
                pred = np.argmax(crf_output, 0)
            else:
                pred = np.argmax(output, axis=0)

            gt = Image.open(os.path.join(args.gt_path, i.strip() + '.png'))
            gt = np.asarray(gt)
            seg_evaluator.add_batch(gt, pred)

            save_path = os.path.join(args.save_path, i.strip() + '.png')
            cv2.imwrite(save_path, pred.astype(np.uint8))

            if args.save_path_c:
                out = pred.astype(np.uint8)
                out = Image.fromarray(out, mode='P')
                out.putpalette(palette)
                out_name = os.path.join(args.save_path_c, i.strip() + '.png')
                out.save(out_name)

        IoU, mIoU = seg_evaluator.Mean_Intersection_over_Union()

        str_format = "{:<15s}\t{:<15.2%}"
        filename = os.path.join(args.save_path, 'result.txt')
        with open(filename, 'w') as f:
            for k in range(args.num_classes):
                print(str_format.format(classes[k], IoU[k]))
                f.write('class {:2d} {:12} IU {:.3f}'.format(k, classes[k], IoU[k]) + '\n')
            f.write('mIoU = {:.3f}'.format(mIoU) + '\n')
        print(f'mIoU={mIoU:.3f}')