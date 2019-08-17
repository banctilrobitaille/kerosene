from typing import Union, List, Tuple


def on_single_device(devices: Union[List, Tuple]):
    return len(devices) == 1
