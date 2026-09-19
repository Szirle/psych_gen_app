# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Device selection helpers that prefer CUDA, then MPS, then CPU."""

import time

import torch

#----------------------------------------------------------------------------

def get_device(pref=None):
    if pref is not None:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

#----------------------------------------------------------------------------

def synchronize(device=None):
    if device is None:
        device = get_device()
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()

#----------------------------------------------------------------------------

def pin_memory_supported(device=None):
    if device is None:
        device = get_device()
    return torch.device(device).type == 'cuda'

#----------------------------------------------------------------------------

class DeviceTimer:
    """Elapsed-time helper that uses CUDA events when available, else wall clocks."""

    def __init__(self, device):
        self.device = torch.device(device)
        self._use_cuda = self.device.type == 'cuda'
        self._start_event = None
        self._end_event = None
        self._t0 = None
        self._t1 = None
        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self._use_cuda:
            self._start_event.record(torch.cuda.current_stream(self.device))
            return
        synchronize(self.device)
        self._t0 = time.perf_counter()

    def stop(self):
        if self._use_cuda:
            self._end_event.record(torch.cuda.current_stream(self.device))
            return
        synchronize(self.device)
        self._t1 = time.perf_counter()

    def elapsed_seconds(self):
        if self._use_cuda:
            self._end_event.synchronize()
            return self._start_event.elapsed_time(self._end_event) * 1e-3
        if self._t0 is None or self._t1 is None:
            return 0.0
        return self._t1 - self._t0

#----------------------------------------------------------------------------
