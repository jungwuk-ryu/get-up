# MacBook에서 pytorch를 사용하기 위해 환경을 확인할 때 사용함

import torch

print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

assert torch.backends.mps.is_available(), "MPS 사용 불가"

# MPS(Metal Performance Shaders)
# Metal사용해서 하드웨어 가속
device = torch.device("mps")

model = torch.nn.Linear(1024, 256).to(device) # 1024개를 256개로 with random weight and bias
x = torch.randn(64, 1024, device=device) # random Normal Distribution(정규분포) 64x1024 to mps
# ArrayList<Data<size=1024>, size=64> 같은 느낌
# List<ECG<size=1024>, size=64>

y = model(x)
loss = y.square().mean() # **2 , avg()
loss.backward()

print("Device:", y.device)
print("Backward:", model.weight.grad is not None)

print(
    "GPU allocated:",
    round(torch.mps.current_allocated_memory() / 1024**2, 2),
    "MiB"
)

print(
    "GPU recommended memory:",
    round(torch.mps.recommended_max_memory() / 1024**3, 2),
    "GiB"
)

# PyTorch: 2.15.0
# MPS available: True
# Device: mps:0
# Backward: True
# GPU allocated: 2.31 MiB
# GPU recommended memory: 17.76 GiB