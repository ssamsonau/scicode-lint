# torch.sigmoid

torch.sigmoid(input, *, out=None) → Tensor

Alias for torch.special.expit().

## Description

Applies the element-wise function:

    sigmoid(x) = 1 / (1 + exp(-x))

The sigmoid function is a smooth, S-shaped (sigmoidal) curve that maps any
real-valued number into the range (0, 1). It is commonly used in machine
learning for:

- Binary classification (as the output layer activation)
- Probability estimation
- Gating mechanisms in RNNs and LSTMs
- Attention mechanisms

## Mathematical Properties

The sigmoid function has several useful mathematical properties:

1. **Range**: The output is always between 0 and 1
2. **Symmetry**: sigmoid(-x) = 1 - sigmoid(x)
3. **Derivative**: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
4. **Monotonic**: The function is strictly increasing

## Parameters

- **input** (Tensor) – the input tensor containing any real numbers.
- **out** (Tensor, optional) – the output tensor. If provided, the result
  will be written to this tensor.

## Returns

A tensor with the sigmoid applied element-wise. The output has the same
shape as the input tensor.

## Example

```python
>>> import torch
>>> x = torch.tensor([0.0, 1.0, 2.0])
>>> torch.sigmoid(x)
tensor([0.5000, 0.7311, 0.8808])

>>> # Negative values
>>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
>>> torch.sigmoid(x)
tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])

>>> # With output tensor
>>> out = torch.empty(3)
>>> torch.sigmoid(torch.tensor([1.0, 2.0, 3.0]), out=out)
>>> out
tensor([0.7311, 0.8808, 0.9526])
```

## Notes

- For numerical stability, use `torch.nn.BCEWithLogitsLoss` instead of
  applying sigmoid followed by `torch.nn.BCELoss`.
- The sigmoid function can cause vanishing gradients for very large or
  very small inputs.

## See Also

- torch.special.expit() - The underlying function.
- torch.nn.Sigmoid - Module wrapper for this function.
- torch.nn.functional.sigmoid - Functional interface.

---

Skip to main content | Back to top | Rate this Page ★★★★★

Docs Access comprehensive developer documentation for PyTorch View Docs
Tutorials Get in-depth tutorials for beginners View Tutorials
Resources Find development resources View Resources

To analyze traffic and optimize your experience, we serve cookies on this site.
By clicking or navigating, you agree to allow our usage of cookies.
Learn more: Cookies Policy.

© 2024 PyTorch Foundation. All rights reserved.
Terms of Service | Privacy Policy | Cookie Settings
