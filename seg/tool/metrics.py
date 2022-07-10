import numpy as np


class Evaluator(object):
    def __init__(self, num_class, ignore=False):
        self.num_class = num_class
        self.ignore = ignore
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Precision_Recall(self):
        precision = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=0) + 1e-5)
        recall = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + 1e-5)
        if self.ignore:
            mp = np.nanmean(precision[:-1])
            mr = np.nanmean(recall[:-1])
            return precision, recall, mp, mr
        else:
            mp = np.nanmean(precision)
            mr = np.nanmean(recall)
            return precision, recall, mp, mr

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        if self.ignore:
            Acc = np.nanmean(Acc[:-1])
        else:
            Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        if self.ignore:
            MIoU = np.nanmean(IoU[:-1])
            return IoU[:-1], MIoU
        else:
            MIoU = np.nanmean(IoU)
            return IoU, MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, "gt: {} pred: {}".format(gt_image.shape, pre_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




