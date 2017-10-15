import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from roi_pooling import RoIPoolFunction


def roi_pooling1(input, rois, size=(7, 7), spatial_scale=1.0):
    F = RoIPoolFunction(size[0], size[1], spatial_scale)
    output = F(input, rois)
    if has_backward:
        F.backward(output.data.clone())
    return output


def roi_pooling2(input, rois, size=(7, 7), spatial_scale=1.0):
    assert rois.dim() == 2
    assert rois.size(1) == 5
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
        output.append(F.adaptive_max_pool2d(im, size))

    output = torch.cat(output, 0)
    if has_backward:
        output.backward(output.data.clone())
    return output


if __name__ == '__main__':
    batch_size = [8, 64, 256]
    size = [8, 64, 256]
    num_rois = [10, 100, 200]
    T = 100
    cuda = True
    has_backward = True
    assert len(batch_size) == len(size)
    assert len(batch_size) == len(num_rois)

    print('use_cuda: {}, has_backward: {}'.format(cuda, has_backward))
    for i in range(len(batch_size)):
        x = Variable(torch.rand((batch_size[i], 3, size[i], size[i])))
        rois = Variable(torch.rand((num_rois[i], 5)))
        rois[:, 0] = rois[:, 0] * batch_size[i]
        rois[:, 1:] = rois[:, 1:] * size[i]
        rois = torch.floor(rois)

        if cuda:
            x = x.cuda()
            rois = rois.cuda()

        for f, foo in enumerate([roi_pooling1, roi_pooling2]):
            start = time.time()
            for t in range(T):
                output = roi_pooling1(x, rois)
            print('method{}: {}, batch_size: {}, size: {}, num_rois: {}'.format(f, time.time() - start,
                                                                                batch_size[i],
                                                                                size[i],
                                                                                num_rois[i]))
