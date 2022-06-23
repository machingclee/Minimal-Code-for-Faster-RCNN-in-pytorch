import torch


def random_choice(arr, length):
    perm_index = torch.randperm(len(arr))
    idxes = perm_index[:int(length)]
    return arr[idxes]
