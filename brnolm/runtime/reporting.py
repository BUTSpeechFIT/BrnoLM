import pathlib
import torch
import math


class ValidationWatcher:
    def __init__(self, val_fn, initial_val_loss, freq_in_tokens, workdir, lm):
        self.val_losses = [initial_val_loss]
        self.validation_fn = val_fn
        self.lm = lm

        pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)

        assert(freq_in_tokens > 0)
        self.freq = freq_in_tokens

        self.report_fn = '{}/validation-report.txt'.format(workdir)
        self.best_model_fn = '{}/best.lm'.format(workdir)
        self.latest_model_fn = '{}/latest.lm'.format(workdir)

        self.running_loss = 0.0
        self.running_targets = 0
        self.running_updates = 0

        self.nb_total_updates = 0

    def log_training_update(self, loss, nb_targets):
        self.running_loss += loss
        self.running_targets += nb_targets
        self.running_updates += 1
        self.nb_total_updates += 1

        if self.running_targets > self.freq:
            val_loss = self.run_validation()

            running_ppl = math.exp(self.running_loss / self.running_updates)
            val_ppl = math.exp(val_loss)

            desc = '{} updates: {:.2f} {:.2f} {:.3f}\n'.format(
                self.nb_total_updates, running_ppl, val_ppl, val_ppl - running_ppl
            )
            with open(self.report_fn, 'a') as f:
                f.write(desc)

            torch.save(self.lm, self.latest_model_fn)
            if min(self.val_losses) == self.val_losses[-1]:
                torch.save(self.lm, self.best_model_fn)

            self.running_loss = 0.0
            self.running_targets = 0
            self.running_updates = 0

    def run_validation(self):
        val_loss = self.validation_fn()
        self.val_losses.append(val_loss)
        return val_loss
