import torch

# The test models are tiny. Spreading each small operation over every CPU core costs more in
# coordination than it saves, so a few threads run the suite many times faster.
torch.set_num_threads(4)
