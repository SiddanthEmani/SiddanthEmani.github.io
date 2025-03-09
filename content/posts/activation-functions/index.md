+++
date = 2025-03-09
title = 'Activation Functions'
tags = ["Machine Learning", "Deep Learning", "Neural Networks", "Activation Functions", "ReLU", "Sigmoid", "Tanh", "Softmax"]
+++
{{< katex >}}

## Sigmoid
{{< chart >}}
type: "line",
data: {
    labels: [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    datasets: [
        {
            label: "Sigmoid",
            data: [0.000045, 0.000123, 0.000335, 0.000911, 0.002472, 0.006693, 0.017986, 0.047426, 0.119203, 0.268941, 0.5, 0.731059, 0.880797, 0.952574, 0.982014, 0.993307, 0.997527, 0.999089, 0.999665, 0.999877, 0.999955],
            borderColor: "rgba(189, 224, 254, 1)"
        },
        {
            label: "Derivative",
            data: [0.000045, 0.000123, 0.000335, 0.000911, 0.002472, 0.00665, 0.01766, 0.0452, 0.1049, 0.196612, 0.25, 0.196612, 0.1049, 0.0452, 0.01766, 0.00665, 0.002472, 0.000911, 0.000335, 0.000123, 0.000045],
            borderColor: "rgba(0, 180, 216, 1)"
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
$$\sigma(x) = \frac{1} {1 + e^{-x}}$$
$$\sigma'(x)=\sigma(x)(1-\sigma(x))$$
Always outputs the number between 0 and 1.

**Use Cases** : Binary Classification ([Logistic Regression](/posts/logistic-regression)), Feed Forward Neural Networks

```python
class CustomSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        output = 1 / (1 + torch.exp(-input))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        output, = ctx.saved_tensors
        grad_input = grad_output * output * (1 - output)
        return grad_input

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return CustomSigmoid.apply(x)
```

---

## Tanh
{{< chart >}}
type: "line",
data: {
    labels: [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    datasets: [
        {
            label: "Tanh",
            data: [
                -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
                -0.9999, -0.9993, -0.9951, -0.9640, -0.7616,
                 0,
                 0.7616,  0.9640,  0.9951,  0.9993,  0.9999,
                 1.0000,  1.0000,  1.0000,  1.0000,  1.0000
            ],
            borderColor: "rgba(189, 224, 254, 1)"
        },
        {
            label: "Derivative",
            data: [
                0, 0, 0, 0, 0,
                0.00018, 0.00134, 0.00990, 0.07197, 0.41970,
                1,
                0.41970, 0.07197, 0.00990, 0.00134, 0.00018,
                0, 0, 0, 0, 0
            ],
            borderColor: "rgba(0, 180, 216, 1)"
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
            ticks: {
                color: 'rgba(255, 255, 255, 0.5)'
            },
            grid: {
                color: 'rgba(255, 255, 255, 0.1)'
            }
        },
        y: {
            ticks: {
                color: 'rgba(255, 255, 255, 0.5)'
            },
            grid: {
                color: 'rgba(255, 255, 255, 0.1)'
            }
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
$$\tanh=\frac{sinh}{\cosh}=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$$
$$\tanh'(x)=1-\tanh^2(x)$$
Always outputs the number between -1 and 1

**Use Cases** : Used in hidden layers since it is zero-centered and the gradients are stronger.

```python
class CustomTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        exp_x = torch.exp(input)
        exp_neg_x = torch.exp(-input)
        tanh = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
        ctx.save_for_backward(tanh)
        return tanh

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        tanh, = ctx.saved_tensors
        grad_input = grad_output * (1 - tanh ** 2)
        return grad_input

def tanh(x: torch.Tensor) -> torch.Tensor:
    return CustomTanh.apply(x)
```

---

## ReLU
{{< chart >}}
type: "line",
data: {
    labels: [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    datasets: [
        {
            label: "ReLU",
            data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            borderColor: "rgba(189, 224, 254, 1)"
        },
        {
            label: "Derivative",
            data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            borderColor: "rgba(0, 180, 216, 1)"
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

$$Relu(x)=max(0,x)$$ 
$$\text{relu}'(x) = \begin{cases} 1 & \text{if } x > 0, \\ 0 & \text{otherwise} \end{cases}$$

Replaces every negative number to zero.

**Use Cases**: Deep Learning models, especially in hidden layers.

Training with saturating non-linearities like [Sigmoid](#sigmoid) and [Tanh](#tanh) is slower than using non-saturating non-linearities like [ReLU](#relu).

There is a non-differentiable kink at x=0.

```python
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

def relu(x: torch.Tensor) -> torch.Tensor:
    return CustomReLU.apply(x)
```

---

## Softplus
{{< chart >}}
type: "line",
data: {
    labels: [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    datasets: [
        {
            label: "Softplus",
            data: [0.000045, 0.000123, 0.000335, 0.000911, 0.002472, 0.006715, 0.01815, 0.04859, 0.12693, 0.31326, 0.693147, 1.313261, 2.126928, 3.048587, 4.018149, 5.006715, 6.002475, 7.000911, 8.000335, 9.000123, 10.000045],
            borderColor: "rgba(189, 224, 254, 1)"
        },
        {
            label: "Derivative",
            data: [0.000045, 0.000123, 0.000335, 0.000911, 0.002472, 0.006693, 0.017986, 0.047426, 0.119203, 0.268941, 0.5, 0.731059, 0.880797, 0.952574, 0.982014, 0.993307, 0.997527, 0.999089, 0.999665, 0.999877, 0.999955],
            borderColor: "rgba(0, 180, 216, 1)"
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
$$softplus(x)=\log(1+e^x)$$
A smooth approximation of [ReLU](#relu).
Larger negative values are close to zero. Positive values behave almost linearly.

Key benefit is it is continuously differentiable everywhere and is equal to [Sigmoid](#sigmoid)
$$\frac{d}{dx}softplus(x)=\frac{1}{1+e^{-x}}=\sigma(x)$$
This is better for optimization during training since it avoids the non-differentiable kink at \\(x=0\\) present in ReLU.

```python
class CustomSoftplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        softplus = torch.log(1 + torch.exp(input))
        ctx.save_for_backward(input)
        return softplus

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = grad_output * (1 / (1 + torch.exp(-input)))
        return grad_input

def softplus(x: torch.Tensor) -> torch.Tensor:
    return CustomSoftplus.apply(x)
```

---

## Argmax
{{< chart >}}
type: "bar",
data: {
    labels: ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"],
    datasets: [
        {
            label: "Input Logits",
            data: [1.2, 3.4, 2.1, 0.8, 4.5],
            backgroundColor: "rgba(0, 180, 216, 1)"
        },
        {
            label: "Argmax",
            data: [0, 0, 0, 0, 1],  // Class 4 has the maximum value
            backgroundColor: "rgba(189, 224, 254, 1)"
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
$$\text{argmax}_xf(x)=x^* \ \ where \ \ f(x^*)\geq f(x)\ \ \ \forall \ x$$

Returns the input value at which a function reaches the maximum.

Since it’s a discrete selection operation, it is not differentiable in the conventional sense.

**Use Cases**: Used in the output layer of a neural network for multi-class classification tasks.

```python
class CustomArgmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int) -> torch.Tensor:
        # Find the maximum value's index along the specified dimension
        max_indices = input.max(dim=dim)[1]
        ctx.save_for_backward(input, max_indices, dim)
        return max_indices

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Non-differentiable, so return zero gradient
        input, _, _ = ctx.saved_tensors
        return torch.zeros_like(input), None  # None for dim

def argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return CustomArgmax.apply(x, dim)
```

---

## Softmax
{{< chart >}}
type: "bar",
data: {
    labels: ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"],
    datasets: [
        {
            label: "Input Logits",
            data: [1.2, 3.4, 2.1, 0.8, 4.5],
            backgroundColor: "rgba(0, 180, 216, 1)"
        },
        {
            label: "Softmax Probabilities",
            data: [0.027, 0.244, 0.066, 0.018, 0.645],  // Computed using Softmax formula
            backgroundColor: "rgba(189, 224, 254, 1)"
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
$$\sigma(z_{i})=\frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}}\ \ \ for\ i=1,2,\ldots K$$
Converts a vector of K real numbers into a probability distribution [0 to 1] of K possible outcomes.

It preserves the order or ranking of the original input values.

**Derivative (Jacobian Matrix):**
\\(\frac{d\sigma(z_{i})}{dz_{j}}=\sigma(z_{i})(\delta_{ij}-\sigma(z_{j}))\\) where \\(\delta_{ij}\\) is the Kronecker delta (1 if \\(i=j\\) and 0 otherwise.)

**Use cases :** Multi-class single label classification

```python
class CustomSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int) -> torch.Tensor:
        exp_x = torch.exp(input)
        sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
        softmax = exp_x / sum_exp_x
        ctx.save_for_backward(softmax, dim)
        return softmax

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        softmax, dim = ctx.saved_tensors
        grad_input = softmax * (grad_output - (softmax * grad_output).sum(dim=dim, keepdim=True))
        return grad_input, None  # None for dim

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return CustomSoftmax.apply(x, dim)
```

## Hierarchical Softmax
{{< chart >}}
type: "bar",
data: {
    labels: ["Root to Left", "Left to Class 0", "Left to Class 1"],
    datasets: [
        {
            label: "Sigmoid Probabilities",
            data: [0.731, 0.268, 0.5],  // Example probabilities for target class 0
            backgroundColor: "rgba(0, 180, 216, 1)"
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
Hierarchical Softmax is an efficient alternative to standard softmax for handling large output spaces. It reduces the computational cost from O(N) to O(log N) by organizing the classes into a binary tree. At each internal node, a sigmoid function is applied to determine the traversal path.

Probability is the product of sigmoid outputs along the path to a class.

**Use Cases**: Large output spaces like language modeling.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalSoftmax(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        """
        Initialize the Hierarchical Softmax layer.
        
        Args:
            input_dim (int): Dimension of the input features.
            num_classes (int): Number of output classes.
        """
        super(HierarchicalSoftmax, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        # Number of internal nodes in a balanced binary tree
        self.num_internal = num_classes - 1
        # Learnable parameters for internal nodes
        self.weights = nn.Parameter(torch.randn(self.num_internal, input_dim))
        self.biases = nn.Parameter(torch.zeros(self.num_internal))
        # Precompute tree structure and paths
        self.tree, self.paths = self._build_tree_and_paths(num_classes)

    def _build_tree_and_paths(self, num_classes: int):
        """
        Build a balanced binary tree and precompute paths for each class.
        
        Args:
            num_classes (int): Number of leaf nodes (classes).
        
        Returns:
            tuple: (tree structure, paths dictionary)
        """
        tree = {}
        paths = {}
        # Construct the tree (0-based indexing for internal nodes)
        for i in range(self.num_internal):
            left = 2 * i + 1
            right = 2 * i + 2
            tree[i] = [left, right] if right < num_classes else [left]
        
        # Precompute paths for each class
        for class_idx in range(num_classes):
            path = []
            node = class_idx + self.num_internal  # Leaf node index
            while node > 0:
                parent = (node - 1) // 2
                if tree[parent][0] == node:
                    path.append((parent, 0))  # Left child
                else:
                    path.append((parent, 1))  # Right child
                node = parent
            paths[class_idx] = list(reversed(path))  # Root-to-leaf path
        return tree, paths

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities for a batch of inputs and targets.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            targets (torch.Tensor): Target classes of shape (batch_size,).
        
        Returns:
            torch.Tensor: Log probabilities of shape (batch_size,).
        """
        batch_size = x.size(0)
        log_probs = torch.zeros(batch_size, device=x.device)

        for i in range(batch_size):
            # Get precomputed path for the target class
            path = self.paths[targets[i].item()]
            prob = 1.0
            # Compute probability along the path
            for node, direction in path:
                logit = torch.matmul(x[i], self.weights[node]) + self.biases[node]
                sigmoid = torch.sigmoid(logit)
                prob *= sigmoid if direction == 0 else (1 - sigmoid)
            # Compute log probability with numerical stability
            log_probs[i] = torch.log(prob + 1e-10)
        
        return log_probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class indices for a batch of inputs.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Predicted class indices of shape (batch_size,).
        """
        batch_size = x.size(0)
        predictions = []
        
        for i in range(batch_size):
            node = 0  # Start at root
            while node < self.num_internal:
                logit = torch.matmul(x[i], self.weights[node]) + self.biases[node]
                sigmoid = torch.sigmoid(logit)
                children = self.tree[node]
                # Traverse left if sigmoid > 0.5, right otherwise
                node = children[0] if sigmoid > 0.5 else children[1] if len(children) > 1 else children[0]
            predictions.append(node - self.num_internal)  # Convert to class index
        
        return torch.tensor(predictions, device=x.device)
```