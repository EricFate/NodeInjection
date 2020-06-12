import torch
from torch.nn import Module


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    # if torch.cuda.is_available():
    return (pred == label).float() \
        .mean().item()
    # .type(torch.cuda.FloatTensor)\
    # else:
    #     return (pred == label).type(torch.FloatTensor).mean().item()


def freeze_model(model: Module):
    for p in model.parameters():
        p.requires_grad = False
