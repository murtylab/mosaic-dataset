import torch

def convert_dataparallel_state_dict_to_vanilla(
    state_dict: dict
):
    vanilla_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module.") :]
        else:
            new_key = key
        vanilla_state_dict[new_key] = value

    return vanilla_state_dict