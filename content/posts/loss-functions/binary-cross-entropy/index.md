+++
date = 2025-03-05
title = 'Binary Cross-Entropy Loss'
tags = ["Machine Learning", "Binary Classification", "Loss Functions", "Supervised Learning"]
+++
{{< katex >}}

## Introduction
{{< chart >}}
type: "line",
data: {
    labels: [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    datasets: [
        {
            label: "BCE Loss (target=0)",
            data: [0.000045, 0.000123, 0.000335, 0.000911, 0.002472, 0.006693, 0.017986, 0.047426, 0.119203, 0.268941, 0.693147, 1.313261, 2.126928, 3.048587, 4.018149, 5.006715, 6.002475, 7.000911, 8.000335, 9.000123, 10.000045],  // Placeholder: replace with computed values
            borderColor: "rgba(0, 180, 216, 1)"
        },
        {
            label: "BCE Loss (target=1)",
            data: [10.000045, 9.000123, 8.000335, 7.000911, 6.002475, 5.006715, 4.018149, 3.048587, 2.126928, 1.313261, 0.693147, 0.268941, 0.119203, 0.047426, 0.017986, 0.006693, 0.002472, 0.000911, 0.000335, 0.000123, 0.000045],  // Placeholder: replace with computed values
            borderColor: "rgba(189, 224, 254, 1)"
        }
    ]
},
options: {
    plugins: {
        legend: {
            labels: {
                color: "white"
            }
        }
    },
    scales: {
        x: { 
            ticks: { color: 'rgba(255, 255, 255, 0.5)' },
            grid: { color: 'rgba(255, 255, 255, 0.1)' }
        },
        y: { 
            ticks: { color: 'rgba(255, 255, 255, 0.5)' },
            grid: { color: 'rgba(255, 255, 255, 0.1)' }
        }
    },
    layout: {
        padding: {
            left: 10,
            right: 10,
            top: 10,
            bottom: 10
        }
    }
}
{{< /chart >}}
Binary Cross-Entropy (BCE) loss is a cornerstone of binary classification tasks in machine learning. However, its standard implementation can encounter numerical instability when dealing with very large or small logits. This post walks through the implementation of a numerically stable BCE loss function in PyTorch, ensuring robustness during model training.

## Mathematical Background
The standard BCE loss for a single sample is defined as:


$$\text{BCE}(z, y) = - \left[ y \cdot \log(p) + (1 - y) \cdot \log(1 - p) \right]$$


where \\( p = \sigma(z) = \frac{1}{1 + e^{-z}} \\) is the sigmoid of the logit \\( z \\), and \\( y \\) is the true label (0 or 1). 

### Underflow
Direct computation of \\( p \\) can lead to overflow or underflow for large \\( |z| \\). A numerically stable alternative is:

$$\text{BCE}(z, y) = \max(z, 0) - y \cdot z + \log(1 + e^{-|z|})$$

This formulation avoids computing \\( \sigma(z) \\) directly, mitigating numerical issues.

## Code
```python
def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable binary cross-entropy loss.
    :param logits: Raw model outputs (logits), shape (batch_size,) or (batch_size, 1)
    :param targets: Ground truth labels (0 or 1), same shape as logits
    :return: Mean loss over the batch
    """
    # Ensure logits are 1D
    logits = logits.squeeze()
    # Compute stable BCE: max(logits, 0) - logits * targets + log(1 + exp(-abs(logits)))
    loss = torch.maximum(logits, torch.zeros_like(logits)) - logits * targets + torch.log1p(torch.exp(-torch.abs(logits)))
    return loss.mean()
```

The implementation of the BCE loss function is crucial for understanding how to handle numerical stability during training :
- **Input Handling**: The `squeeze()` operation ensures logits are a 1D tensor, accommodating varying input shapes.
- **Stable Computation**: The formula leverages `torch.maximum` and `torch.log1p` (log(1 + x) for small x) to prevent overflow/underflow.
- **Batch Averaging**: The mean loss is returned, suitable for optimization.
This implementation is critical for training models where logits may vary widely in magnitude, ensuring numerical reliability.

- **Applications**: Binary classification tasks, including emotion detection, fraud detection, and medical diagnosis.
- **Performance Considerations**: While this implementation is stable, batch size and input distribution can significantly affect training dynamics. 