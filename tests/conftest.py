"""Test-suite settings: two CPU threads, the machine budget (multi-threaded
LayerNorm backward on CPU has crashed with 4+ threads in this environment)."""
import torch

torch.set_num_threads(2)
