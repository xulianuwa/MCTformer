import numpy as np
import torch
import cv2
import random
import os
import PIL.Image as Image


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line.strip())
    return img_list


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def scale_gt(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims, interpolation=cv2.INTER_NEAREST).astype(float)


def RandomCrop(imgarr, cropsize):
    h, w, c = imgarr.shape

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    img_container = np.zeros((cropsize, cropsize, imgarr.shape[-1]), np.float32)

    cropping = np.zeros((cropsize, cropsize), np.bool)

    img_container[cont_top:cont_top + ch, cont_left:cont_left + cw] = \
        imgarr[img_top:img_top + ch, img_left:img_left + cw]
    cropping[cont_top:cont_top + ch, cont_left:cont_left + cw] = 1

    return img_container, cropping



def crop(img_temp, dim, new_p=True, h_p=0, w_p=0):
    h = img_temp.shape[0]
    w = img_temp.shape[1]
    trig_h = trig_w = False
    if (h > dim):
        if (new_p):
            h_p = int(random.uniform(0, 1) * (h - dim))
        if len(img_temp.shape) == 2:
            img_temp = img_temp[h_p:h_p + dim, :]
        else:
            img_temp = img_temp[h_p:h_p + dim, :, :]
    elif (h < dim):
        trig_h = True
    if (w > dim):
        if (new_p):
            w_p = int(random.uniform(0, 1) * (w - dim))
        if len(img_temp.shape) == 2:
            img_temp = img_temp[:, w_p:w_p + dim]
        else:
            img_temp = img_temp[:, w_p:w_p + dim, :]
    elif (w < dim):
        trig_w = True
    if (trig_h or trig_w):
        if len(img_temp.shape) == 2:
            pad = np.zeros((dim, dim))
            pad[:img_temp.shape[0], :img_temp.shape[1]] = img_temp
        else:
            pad = np.zeros((dim, dim, 3))
            pad[:img_temp.shape[0], :img_temp.shape[1], :] = img_temp
        return (pad, h_p, w_p)
    else:
        return (img_temp, h_p, w_p)


def get_data_from_chunk(chunk, args):
    img_path = args.img_path

    scale = np.random.uniform(0.7, 1.3)
    dim = args.crop_size
    images = np.zeros((dim, dim, 3, len(chunk)))
    ori_images = np.zeros((dim, dim, 3, len(chunk)), dtype=np.uint8)
    gt_images = np.zeros((dim, dim, len(chunk)))

    img_names = []

    for i, pieces in enumerate(chunk):
        piece = pieces.replace('.jpg','').strip()
        img_names.append(piece)
        flip_p = np.random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg'))
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float)

        gt_temp = np.asarray(Image.open(os.path.join(args.seg_pgt_path, piece + '.png')))
        img_temp = scale_im(img_temp, scale)
        gt_temp = scale_gt(gt_temp, scale)
        img_temp = flip(img_temp, flip_p)
        gt_temp = flip(gt_temp, flip_p)

        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

        img_temp, img_temp_h_p, img_temp_w_p = crop(img_temp, dim)
        gt_temp = crop(gt_temp, dim, False, img_temp_h_p, img_temp_w_p)[0]
        gt_images[:, :, i] = gt_temp[:, :]

        ori_temp = np.zeros_like(img_temp)
        ori_temp[:, :, 0] = (img_temp[:, :, 0] * 0.229 + 0.485) * 255.
        ori_temp[:, :, 1] = (img_temp[:, :, 1] * 0.224 + 0.456) * 255.
        ori_temp[:, :, 2] = (img_temp[:, :, 2] * 0.225 + 0.406) * 255.
        ori_images[:, :, :, i] = ori_temp.astype(np.uint8)

        images[:, :, :, i] = img_temp

    images = images.transpose((3, 2, 0, 1))
    ori_images = ori_images.transpose((3, 2, 0, 1))
    gt_images = gt_images.transpose((2, 0, 1))
    gt_images = torch.from_numpy(gt_images).float()
    images = torch.from_numpy(images).float()
    return images, ori_images, gt_images, img_names
