# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir, update_freq):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)
        self.update_freq = update_freq
        self.step = 0

    def next_step(self):
        self.step += 1

    def logging_step(self):
        return (self.step+1) % self.update_freq == 0

    def scalar_summary(self, tag, value, enforce=False):
        if not enforce and not self.logging_step():
            return

        self.writer.add_scalar(tag, value, self.step)

    def hierarchical_scalar_summary(self, master_tag, tag, value, enforce=False):
        if not enforce and not self.logging_step():
            return

        self.writer.add_scalars(master_tag, {tag: value}, self.step)

    def histo_summary(self, tag, values, bins=1000, enforce=False):
        if not enforce and not self.logging_step():
            return

        self.writer.add_histogram(tag, values, self.step, max_bins=bins)
