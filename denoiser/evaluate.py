# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import sys

from pesq import pesq
from pystoi import stoi
import torch
import numpy as np

from .data import NoisyCleanSet
from .enhance import add_flags, get_estimate
from . import distrib, pretrained
from .utils import bold, LogProgress

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
        'denoiser.evaluate',
        description='Speech enhancement using Demucs - Evaluate model performance')
add_flags(parser)
parser.add_argument('--data_dir', help='directory including noisy.json and clean.json files')
parser.add_argument('--matching', default="sort", help='set this to dns for the dns dataset.')
parser.add_argument('--no_pesq', action="store_false", dest="pesq", default=True,
                    help="Don't compute PESQ.")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="More loggging")


def evaluate(args, model=None, data_loader=None):
    total_pesq = 0
    total_stoi = 0
    total_sdr = 0
    total_cnt = 0
    updates = 5

    # Load model
    if not model:
        model = pretrained.get_model(args).to(args.device)
    model.eval()

    # Load data
    if data_loader is None:
        dataset = NoisyCleanSet(args.data_dir,
                                matching=args.matching, sample_rate=model.sample_rate)
        data_loader = distrib.loader(dataset, batch_size=1, num_workers=2)
    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                noisy, clean = [x.to(args.device) for x in data]
                # If device is CPU, we do parallel evaluation in each CPU worker.
                if args.device == 'cpu':
                    pendings.append(
                        pool.submit(_estimate_and_run_metrics, clean, model, noisy, args))
                else:
                    estimate = get_estimate(model, noisy, args)
                    estimate = estimate.cpu()
                    clean = clean.cpu()
                    pendings.append(
                        pool.submit(_run_metrics, clean, estimate, args, model.sample_rate))
                total_cnt += clean.shape[0]
        for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
            pesq_i, stoi_i, sdr_i = pending.result()
            total_pesq += pesq_i
            total_stoi += stoi_i
            total_sdr += sdr_i

    metrics = [total_pesq, total_stoi, total_sdr]
    pesq, stoi, sdr = distrib.average([m/total_cnt for m in metrics], total_cnt)
    logger.info(bold(f'Test set performance:PESQ={pesq}, STOI={stoi}, SDR={sdr}.'))
    return pesq, stoi


def _estimate_and_run_metrics(clean, model, noisy, args):
    estimate = get_estimate(model, noisy, args)
    return _run_metrics(clean, estimate, args, sr=model.sample_rate)


def _run_metrics(clean, estimate, args, sr):
    estimate = estimate.numpy()[:, 0]
    clean = clean.numpy()[:, 0]
    if args.pesq:
        pesq_i = get_pesq(clean, estimate, sr=sr)
    else:
        pesq_i = 0
    stoi_i = get_stoi(clean, estimate, sr=sr)
    sdr_i = get_sdr(clean, estimate)
    return pesq_i, stoi_i, sdr_i


def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    return pesq_val


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val

def get_sdr(clean_signal, estimated_signal):
    """
    Calculate the Signal-to-Distortion Ratio (SDR).

    Parameters:
    - clean_signal: numpy array, the original clean audio signal.
    - estimated_signal: numpy array, the estimated audio signal.

    Returns:
    - SDR: float, the calculated SDR value in dB.
    """
    sdr = 0
    for i in range(len(clean_signal)):
        # Ensure the signals are aligned in length
        min_len = min(len(clean_signal[i]), len(estimated_signal[i]))
        clean_signal = clean_signal[i][:min_len]
        estimated_signal = estimated_signal[i][:min_len]

        # Calculate signal power and error power
        signal_power = np.sum(clean_signal[i] ** 2)
        error_power = np.sum((clean_signal[i] - estimated_signal[i]) ** 2)

        # Calculate SDR
        if signal_power < 1e-5:
            #print('signal_power < 1e-5')
            continue
        
        sdr += 10 * np.log10(signal_power / (error_power + 1e-10))
        #print('sdr: ',sdr)

    return sdr

def main():
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    pesq, stoi = evaluate(args)
    json.dump({'pesq': pesq, 'stoi': stoi}, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
