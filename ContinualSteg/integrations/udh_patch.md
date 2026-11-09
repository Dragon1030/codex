# UDH integration contract

The standalone backend in this repository is a compact UDH-style model. To use
the exact uploaded UDH code, add an optional `prompt` argument to
`models/HidingUNet.py` and `models/RevealNet.py`.

The first safe insertion point is the bottleneck feature. Apply a
`SpatialAdapter(256, bottleneck)` and a prompt bias there, then pass the task id
from `main_continual.py`. Keep the original `main.py` unchanged.

The training order is:

1. Load the completed UDH baseline checkpoint and train task 0.
2. Freeze the HidingUNet/RevealNet backbone and task 0 adapter.
3. Allocate task 1's adapter and train it on the new attack domain.
4. Repeat for later domains, storing a stage checkpoint and a small replay set.

The exact UDH input pair remains `(cover, secret)`, the hidden output is
`container`, and `RevealNet(container)` is the recovered secret. Existing UDH
noise layers can replace the differentiable attack approximations in
`continual_steg/tasks.py`.
