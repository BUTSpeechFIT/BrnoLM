class ModelStatistics:
    def __init__(self, model):
        self.model = model

    def total_nb_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def nb_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def nb_nontrainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

    def __str__(self):
        torch_desc = f'{self.model}\n'
        nb_params_desc = f'Total number of parameters: {self.total_nb_params()/1000000:.2f}M\n'
        nb_trainable_desc = f'Number of trainable parameters: {self.nb_trainable_params()/1000000:.2f}M\n'
        nb_nontrainable_desc = f'Number of nontrainable parameters: {self.nb_nontrainable_params()/1000000:.2f}M\n'
        return torch_desc + nb_params_desc + nb_trainable_desc + nb_nontrainable_desc
