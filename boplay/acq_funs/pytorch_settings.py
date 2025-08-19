import torch as pt

PT_DTYPE = pt.float32
PT_DEVICE = pt.device("cuda" if pt.cuda.is_available() else "cpu")