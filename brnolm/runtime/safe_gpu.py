import os
import time
import subprocess
import fcntl
import logging


LOCK_FILENAME = '/tmp/gpu-lock-magic-ibenes-RaNdOM'


def setup_cuda_visible_devices():
    free_gpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = free_gpu.decode().strip()


class SafeLocker:
    def __init__(self, fd):
        self._fd = fd

    def __enter__(self):
        fcntl.lockf(self._fd, fcntl.LOCK_EX)

    def __exit__(self, type, value, traceback):
        fcntl.lockf(self._fd, fcntl.LOCK_UN)


class GPUOwner:
    def __init__(self, placeholder_fn, logger=None, debug_sleep=0.0):
        if logger is None:
            logger = logging

        with open(LOCK_FILENAME, 'w') as f:
            logger.info(f'acquiring lock')

            with SafeLocker(f):
                logger.info(f'lock acquired')
                setup_cuda_visible_devices()

                logger.info(f"Got CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
                time.sleep(debug_sleep)

                try:
                    self.placeholder = placeholder_fn()
                except RuntimeError:
                    logger.error(f'failed to acquire placeholder')
                    raise
            logger.info(f'lock released')
