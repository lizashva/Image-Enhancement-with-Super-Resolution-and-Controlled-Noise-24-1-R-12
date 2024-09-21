#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn

from function.N2N.datasets import load_dataset
from function.N2N.noise2noise import Noise2Noise

from argparse import ArgumentParser

def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='../data')
    parser.add_argument('--load-ckpt', help='load model checkpoint')
    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'poisson', 'text', 'mc'], default='gaussian', type=str)
    parser.add_argument('-v', '--noise-param', help='noise parameter (e.g. sigma for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=256, type=int)

    return parser.parse_args()


def test_N2N(config):
    """Tests Noise2Noise."""

    # Parse test parameters
    #params = parse_args()
    params = config
    print(os.getcwd())
    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False)
    params.redux = False
    params.clean_targets = True
    test_loader = load_dataset(params.data, 0, params, shuffled=False, single=True, noisy_source=params.noisy_source)
    n2n.load_model(params.load_ckpt)        # The learned parameters for making predictions on new data.
    if params.noisy_source:
        path = n2n.denoise(test_loader, show=params.show_output)
    else:
        path = n2n.test(test_loader, show=params.show_output)
    return path
