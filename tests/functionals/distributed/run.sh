#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --amp-opt-level=O1