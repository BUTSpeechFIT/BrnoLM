import sys
import time
import math

import torch


class BaseLogger():
    def __init__(self, report_period, output_file=sys.stdout):
        self._start_time = time.time()
        self._nb_logs = 0
        self._report_period = report_period
        self._of = output_file
        self._construction_time = time.time()

    def log(self, *args):
        self._log(*args)
        self._nb_logs += 1

        if self._nb_logs % self._report_period == 0:
            self._flush()
            self._reset()
            self._start_time = time.time()

    def nb_updates(self):
        return self._nb_logs

    def time_since_creation(self):
        return time.time() - self._construction_time

    def _flush(self):
        pass

    def _reset(self):
        pass

    def _log(self, *args):
        pass


class InfinityLogger(BaseLogger):
    def __init__(self, epoch, report_period, lr, output_file=sys.stdout):
        super().__init__(report_period, output_file)
        self._running_loss = 0.0
        self._epoch = epoch
        self._lr = lr

    def _log(self, loss):
        self._running_loss += loss

    def _flush(self):
        ms_per_log = (time.time() - self._start_time) * 1000 / self._report_period
        cur_loss = (self._running_loss / self._report_period).item()
        fmt_string = '| epoch {:3d} | {:5d} batches done | lr {:.3e} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}\n'
        line = fmt_string.format(
            self._epoch, self._nb_logs, self._lr,
            ms_per_log, cur_loss, math.exp(cur_loss)
        )
        self._of.write(line)

    def _reset(self):
        self._running_loss = 0.0


class GradLogger(BaseLogger):
    def __init__(self, report_period, named_params, output_file=sys.stdout):
        super().__init__(report_period, output_file)
        self._named_params = list(named_params)

        self._grads = {}
        for name, param in self._named_params:
            self._grads[name] = []

        self._of.write("{}\n".format(" ".join(self._grads)))

    def _log(self):
        for name, param in self._named_params:
            self._grads[name].append(param.grad.abs().mean())

    def _flush(self):
        grad_mavs = []
        for name in self._grads:
            all_grads = torch.stack(self._grads[name])
            grad_mavs.append(all_grads.mean().data.item())

        fmt_string = " ".join("{:.7f}" for _ in grad_mavs) + "\n"
        line = fmt_string.format(*grad_mavs)
        self._of.write(line)

    def _reset(self):
        for name in self._grads:
            self._grads[name] = []


class ProgressLogger():
    def __init__(self, epoch, report_period, lr, nb_updates, output_file=sys.stdout):
        self._start_time = time.time()
        self._nb_logs = 0
        self._running_loss = 0.0
        self._epoch = epoch
        self._report_period = report_period
        self._of = output_file
        self._lr = lr
        self._construction_time = time.time()
        self._nb_updates = nb_updates

    def log(self, loss):
        self._running_loss += loss
        self._nb_logs += 1

        if self._nb_logs % self._report_period == 0:
            self._flush()
            self._reset()

    def time_since_creation(self):
        return time.time() - self._construction_time

    def nb_updates(self):
        return self._nb_updates

    def _flush(self):
        ms_per_log = (time.time() - self._start_time) * 1000 / self._report_period
        cur_loss = (self._running_loss / self._report_period).item()
        fmt_string = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:.3e} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}\n'
        line = fmt_string.format(
            self._epoch, self._nb_logs, self._nb_updates, self._lr,
            ms_per_log, cur_loss, math.exp(cur_loss)
        )
        self._of.write(line)

    def _reset(self):
        self._running_loss = 0.0
        self._start_time = time.time()


class NoneLogger():
    def log(self, *args):
        pass
