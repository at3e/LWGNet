#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torch.nn as nn


def init_obj_field(x, scale):
    _, h, w = x.shape
    xFT_pad = nn.functional.upsample(x.unsqueeze(0), scale_factor=scale, mode='bilinear')
    return xFT_pad.squeeze(0)

def alpha_blending(img1, img2, n_ovlap, direction):
    S1 = img1.shape
    S2 = img2.shape
    img1 = (img1-np.mean(img1))/(np.max(img1)-np.min(img1))
    img2 = (img2-np.mean(img2))/(np.max(img2)-np.min(img2))

    if direction=='horizontal':
        assert S1[0] == S2[0]
        row_img1 = np.ones((1, S1[1]), dtype=np.float64)
        row_img2 = np.ones((1, S2[1]), dtype=np.float64)
        alpha_line1 = np.linspace(1, 0, n_ovlap)
        alpha_line2 = np.linspace(0, 1, n_ovlap)
        row_img1[0,-n_ovlap:] = alpha_line1
        row_img2[0,:n_ovlap] = alpha_line2
        alpha_img1 = np.repeat(row_img1, S1[0], axis=0)
        alpha_img2 = np.repeat(row_img2, S2[0], axis=0)
        alpha_img1 = np.expand_dims(alpha_img1, axis=0)
        alpha_img2 = np.expand_dims(alpha_img2, axis=0)

        img_out1 = np.zeros((S1[0], S1[1] + S2[1] - n_ovlap), dtype=np.float64)
        img_out2 = np.zeros((S1[0], S1[1] + S2[1] - n_ovlap), dtype=np.float64)
        img_out1[:, :S1[1]] = alpha_img1 * img1
        img_out2[:, S1[1]-n_ovlap:] = alpha_img2 * img2

    if direction=='vertical':
        col_img1 = np.ones((S1[0], 1), dtype=np.float64)
        col_img2 = np.ones((S2[0], 1), dtype=np.float64)
        alpha_line1 = np.linspace(1, 0, n_ovlap).T
        alpha_line2 = np.linspace(0, 1, n_ovlap).T
        col_img1[-n_ovlap:, 0] = alpha_line1
        col_img2[:n_ovlap, 0] = alpha_line2
        alpha_img1 = np.repeat(col_img1, S1[1], axis=1)
        alpha_img2 = np.repeat(col_img2, S2[1], axis=1)

        img_out1 = np.zeros((S1[0] + S2[0] - n_ovlap, S1[1]), dtype=np.float64)
        img_out2 = np.zeros((S1[0] + S2[0] - n_ovlap, S2[1]), dtype=np.float64)
        img_out1[:S1[0], :] = alpha_img1 * img1
        img_out2[S1[0]-n_ovlap:, :] = alpha_img2 * img2

    return img_out1+img_out2

def load_module_weights(model, state_dict):
    pre_trained_keys = []
    for key, value in state_dict.items():
        pre_trained_keys.append(key)

    #Load weights and bias of the Conv2d layers only
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            for key in pre_trained_keys:
                if key == name+'.weight' and state_dict[key].shape==layer.weight.shape:
                    layer.weight.data = state_dict[key].cuda()
                elif key == name+'.bias' and state_dict[key].shape==layer.bias.shape:
                    layer.bias.data = state_dict[key].cuda()

    return model


def prepare_empty_dir(dirs, resume=False):
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)

class ExecutionTime:
    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return time.time() - self.start_time



