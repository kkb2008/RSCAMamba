import thop
import torch
from geoseg.models.newPyramidMamba import swinMamba_base, swinMamba_tiny, swinMamba_small, resMamba_34

input_tensor = torch.randn(1, 3, 512, 512)

model = swinMamba_base(10)
input_tensor = input_tensor.to('cuda')
model = model.to('cuda')

flops, params = thop.profile(model, inputs=(input_tensor,))

print(f"FLOPs: {flops / 1e9}G")
print(f"Parameters: {params/1e6}M")
