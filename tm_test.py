import numpy as np 


# print(np.load("data/extra/class-ids-TRAIN.npy", mmap_mode="r"))
# print(np.load("data/extra/entries-TRAIN.npy", mmap_mode="r"))
# print(np.load("data/extra/class-names-TRAIN.npy", mmap_mode="r"))

import torch 

torch.hub.set_dir("./pretrained/")
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitl14.to("cuda")
# print(dinov2_vitl14.load_state_dict(torch.load("pretrained/checkpoints/dinov2_vitl14_pretrain.pth")))

# print(dinov2_vitl14)

dummy_input = torch.rand(2,3,448,448).to("cuda")
print(dinov2_vitl14(dummy_input))