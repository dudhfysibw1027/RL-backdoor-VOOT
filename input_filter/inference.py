import torch
from input_filter.Configs import Config as Configs
from input_filter.models.model import base_Model
import numpy as np

single_sample = np.random.randn(1, 18, 5)  # shape = (1, 18, 5)

def load_trained_model(checkpoint_path, device='cpu'):
    configs = Configs()
    model = base_Model(configs)
    chkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(chkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model, configs


checkpoint_path = "input_filter/checkpoints/mobile_0217_2/ckp_last.pt"
model, configs = load_trained_model(checkpoint_path, device="cuda:0")


def real_time_inference(model, single_sample, device='cpu'):
    """
    single_sample: shape=(1, 18, 5) 的 numpy array or tensor
    回傳: (logits, predicted_label)
    """
    model.eval()

    if isinstance(single_sample, np.ndarray):
        single_sample = torch.tensor(single_sample, dtype=torch.float32)

    single_sample = single_sample.to(device)  # (1, 18, 5)

    with torch.no_grad():
        logits, features = model(single_sample)  # (1, num_classes)
        predicted_label = torch.argmax(logits, dim=1)  # shape=(1,)
    return logits.cpu().numpy(), predicted_label.cpu().numpy()


# pseudo sample
single_logit, single_pred = real_time_inference(model, single_sample, device="cuda:0")
print("Logits:", single_logit)
print("Predicted label:", single_pred)

