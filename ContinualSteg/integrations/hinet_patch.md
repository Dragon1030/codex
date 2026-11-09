# HiNet integration contract

HiNet is invertible, so the task condition must be present in both directions.
Add three zero-initialized linear projections to `INV_block`:

```python
self.prompt_r = nn.Linear(prompt_dim, self.split_len2)
self.prompt_y = nn.Linear(prompt_dim, self.split_len2)
self.prompt_f = nn.Linear(prompt_dim, self.split_len1)
```

For the forward coupling, use:

```python
t2 = self.f(x2) + self.prompt_f(prompt)[:, :, None, None]
y1 = x1 + t2
s1 = self.r(y1) + self.prompt_r(prompt)[:, :, None, None]
t1 = self.y(y1) + self.prompt_y(prompt)[:, :, None, None]
y2 = self.e(s1) * x2 + t1
```

For the reverse coupling, calculate the same conditioned `s1`, `t1`, and `t2`
before inverting. The same prompt must be supplied to `rev=False` and
`rev=True`; otherwise the inverse no longer matches the forward transform.

Change `Hinet.forward`, `Model.forward`, and each `INV_block` call to accept
`prompt=None`. When `prompt is None`, the original HiNet behavior remains
unchanged. Initialize the three prompt projections to zero so loading an old
HiNet checkpoint gives an identity continual adapter.

The continual trainer should call:

```python
container_latent = net(input_img, prompt=prompt)
decoded = net(output_rev, rev=True, prompt=prompt)
```

Use the `ContinualHidingModel` training/checkpoint protocol as the reference for
task adapters, replay and stage metrics.
