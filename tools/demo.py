#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对data/demo下面的所有图片进行faster-rcnn自定义类别检测，假定只有车牌（licencePlate）这一种类别
"""

import matplotlib
matplotlib.use('Agg')

import _init_paths
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe
import os
import sys
import cv2
import argparse

CLASSES = ('__background__',
           'licencePlate')


def vis_detections(im, class_name, dets, ax, thresh):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')


def demo(net, image_name):
    im_file = 'home/zxh/testimg/' + image_name
    im = cv2.imread(im_file)
    scores, boxes = im_detect(net, im)
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)


if __name__ == '__main__':
    prototxt = '/home/zxh/cdcaffe/models/faster_rcnn/test.prototxt'
    caffemodel = '/home/zxh/cdcaffe/models/faster_rcnn/VGG16_faster_rcnn_final.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(0)
    # 如果只是用cpu模式，将上面三行换成 caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)
    im_names = ['1.jpg', '2.jpg', '3.jpg']
    for im_name in im_names:
        demo(net, im_name)
        plt.savafig("/home/zxh/faster_rcnn_results/" + im_name)
