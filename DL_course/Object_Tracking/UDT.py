# -*- coding: utf8 -*-
from __future__ import division

import argparse
import numpy as np
import torch
import sys
import cv2
import time as time
import argparse
from util import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox
from net import DCFNet

parser = argparse.ArgumentParser(description='Select your input sources')
parser.add_argument('--input_source', default='0', type=str, help='Web-cam(Default) :{0}, Others : {your_videos}')
args = parser.parse_args()



class TrackerConfig(object):
    # These are the default hyper-params for DCFNet
    # OTB2013 / AUC(0.665)
    feature_path = 'param.pth'
    crop_sz = 125

    lambda0 = 1e-4
    padding = 2
    output_sigma_factor = 0.1
    interp_factor = 0.01
    num_scale = 3
    scale_step = 1.0275
    scale_factor = scale_step ** (np.arange(num_scale) - num_scale / 2)
    min_scale_factor = 0.2
    max_scale_factor = 5
    scale_penalty = 0.9925
    scale_penalties = scale_penalty ** (np.abs((np.arange(num_scale) - num_scale / 2)))

    net_input_size = [crop_sz, crop_sz]
    net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, net_input_size)
    yf = torch.rfft(torch.Tensor(y).view(1, 1, crop_sz, crop_sz).cuda(), signal_ndim=2)
    cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()

    model = 'param.pth'


def extract_roi(ret, frame, frame_name):
    if not ret:
        print('Frame read failed!')
        sys.exit()

    init_rect = cv2.selectROI(frame_name, frame)
    target_pos, target_sz = rect1_2_cxy_wh(init_rect)

    return target_pos, target_sz


if __name__ == '__main__':
    # Initialize Tracking Network
    config = TrackerConfig()
    net = DCFNet(config)
    net.load_param(config.model)
    net.eval().cuda()

    # Initiate Video Files
    if args.input_source == '0':
        cap = cv2.VideoCapture('/dev/video0')
        print("You Selected Input Soruce : /dev/video0")
    else:
        cap = cv2.VideoCapture(args.input_source)
        print("You Selected Input Soruce :", args.input_source)
    if not cap.isOpened():
        print('Video open failed!')
        sys.exit()

    ret, frame = cap.read()
    target_pos, target_sz = extract_roi(ret, frame, 'frame')

    # crop template and Forward Tracking
    min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
    max_sz = np.minimum(frame.shape[:2], config.max_scale_factor * target_sz)

    window_sz = target_sz * (1 + config.padding)
    bbox = cxy_wh_2_bbox(target_pos, window_sz)
    patch = crop_chw(frame, bbox, config.crop_sz)

    target = patch - config.net_average_image
    net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())

    patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)

    pause = False

    # loop videos
    while True:
        tic = time.time()
        if pause:  # Pause Condition : Press "space bar"
            ret, frame = cap.read()
            target_pos, target_sz = extract_roi(ret, frame, 'frame')
            # crop template and Forward Tracking
            min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
            max_sz = np.minimum(frame.shape[:2], config.max_scale_factor * target_sz)
            window_sz = target_sz * (1 + config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)
            patch = crop_chw(frame, bbox, config.crop_sz)
            target = patch - config.net_average_image
            net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())
            patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)

            pause = False

        ret, frame = cap.read()
        if not ret:
            print('We show final frame')
            sys.exit()

        for i in range(config.num_scale):
            window_sz = target_sz * (config.scale_factor[i] * (1 + config.padding))
            bbox = cxy_wh_2_bbox(target_pos, window_sz)
            patch_crop[i, :] = crop_chw(frame, bbox, config.crop_sz)

        # Backward Tracking
        search = patch_crop - config.net_average_image
        response = net(torch.Tensor(search).cuda())
        peak, idx = torch.max(response.view(config.num_scale, -1), 1)
        peak = peak.data.cpu().numpy() * config.scale_penalties
        idx = idx.data.cpu().numpy()
        best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx[best_scale], config.net_input_size)

        if r_max > config.net_input_size[0] / 2:
            r_max -= config.net_input_size[0]
        if c_max > config.net_input_size[1] / 2:
            c_max -= config.net_input_size[1]
        window_sz = target_sz * (config.scale_factor[best_scale] * (1 + config.padding))

        target_pos += np.array([c_max, r_max]) * window_sz / config.net_input_size
        target_sz = np.minimum(np.maximum(window_sz / (1 + config.padding), min_sz), max_sz)

        window_sz = target_sz * (1 + config.padding)
        bbox = cxy_wh_2_bbox(target_pos, window_sz)
        patch = crop_chw(frame, bbox, config.crop_sz)
        target = patch - config.net_average_image
        net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=config.interp_factor)

        # Show predicted box
        cv2.rectangle(frame, (int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)),
                      (int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)), (0, 255, 0), 3)

        tac = time.time()
        cv2.putText(frame, str(1/(tac-tic))[:6] + "fps", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            pause = True
