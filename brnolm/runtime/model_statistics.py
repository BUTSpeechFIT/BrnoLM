def scaled_int_str(value):
    if value < 1000:
        return f'{value}'
    elif value < 1000000:
        return f'{value/1000:.1f}k'
    else:
        return f'{value/1000000:.1f}M'


class ModelStatistics:
    def __init__(self, model):
        self.model = model

    def total_nb_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def nb_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def trainable_params_breakup(self):
        per_param_desc = (f'{name} {scaled_int_str(p.numel())}\n' for name, p in self.model.named_parameters() if p.requires_grad)
        return ''.join(per_param_desc)

    def nb_nontrainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

    def __str__(self):
        torch_desc = f'{self.model}\n'
        nb_params_desc = f'Total number of parameters: {scaled_int_str(self.total_nb_params())}\n'
        nb_trainable_desc = f'Number of trainable parameters: {scaled_int_str(self.nb_trainable_params())}\n'
        nb_nontrainable_desc = f'Number of nontrainable parameters: {scaled_int_str(self.nb_nontrainable_params())}\n'
        return torch_desc + nb_params_desc + nb_trainable_desc + nb_nontrainable_desc + self.trainable_params_breakup()
