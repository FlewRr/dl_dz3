import torch

def recall_at_k(target: torch.Tensor, predicted: torch.Tensor, k: int) -> float:
    topk_preds = predicted[:, :k]
    target_expanded = target.unsqueeze(-1).expand_as(topk_preds)

    correct = (topk_preds == target_expanded).any(dim=1)

    return correct.float().mean().item()