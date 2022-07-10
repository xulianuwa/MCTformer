import numpy as np

def crf_inference_inf(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=4/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=83/scale_factor, srgb=5, rgbim=np.copy(img_c), compat=3)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))