import torch
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils

def load_model(model_name: str = "gpt2-large", device: str = None):
    if device is None:
        device = utils.get_device()

    model = HookedTransformer.from_pretrained(model_name, device=device)
    torch.set_grad_enabled(False)

    # Replace layer norms with identity as in your notebook
    for layer in model.blocks:
        layer.ln1 = torch.nn.Identity()
        layer.ln2 = torch.nn.Identity()

    return model
