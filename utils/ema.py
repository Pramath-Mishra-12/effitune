import copy
import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self._backup = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * self.decay + (1.0 - self.decay) * model_v)

    def apply_shadow(self, model: torch.nn.Module):
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema.state_dict())

    def restore(self, model: torch.nn.Module):
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None