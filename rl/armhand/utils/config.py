from typing import Dict

from omegaconf import DictConfig


def omegaconf_to_dict(d: DictConfig) -> Dict:
    ret = {}
    for k, v in d.items():
        ret[k] = omegaconf_to_dict(v) if isinstance(v, DictConfig) else v
    return ret


def print_dict(val, nesting: int = -4, start: bool = True):
    if type(val) == dict:
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        print(val)
