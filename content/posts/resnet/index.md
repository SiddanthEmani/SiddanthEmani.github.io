+++
date = 2025-01-09
title = 'What is ResNet?'
tags = ["Deep Learning", "Computer Vision", "ResNet"]
+++
{{< katex >}}

### ResNet Block

![ResNet Architecture](img/ResNet%20Architecture.png)

{{< article link="/posts/resnet/" >}}

- **3x3 Convolution (stride=2)** - Downsamples the image into half the height and width. Also, doubles the number of channels (number of filters are doubled).
    - Sliding a 3x3 kernel/matrix onto the image and dot product is calculated with overlapped images. 3x3 is small enough to capture fine details and more efficient. Can be stacked to capture complex patterns.
- **Batch Norm**
- [ReLU](img/Activation%20Functions#Rectified%20Linear%20Unit%20(ReLU))

### Thought Process

Our goal is to **increase the resolution** of the image. 

![ResNet Process 1](img/ResNet%20Process%201.png)

If the model is trained on a vanilla deep net with both low and high resolution input images, it **will increase our training loss** since the input signal is being lost when its passed through a lot of layers.

![ResNet Process 2](img/ResNet%20Process%202.png)

Our goal is to now **learn the residual** to be added to the image to increase its resolution.

![ResNet Process 3](img/ResNet%20Process%203.png)

The idea is to **add the input image to our residual** at the end of our block.
Two problems arise:
1. Dimension mismatch (When the input and residual have different dimensions, we cannot add)
2. A steady stream of input data across the network

![ResNet Process 4](img/ResNet%20Process%204.png)

So we add residual connections to handle these. And we treat the network as a **series of residual blocks** instead of layers.

![ResNet Process 5](img/ResNet%20Process%205.png)

Each block **does not take penalty** from the loss function since it can output the same identity function.

This enables very deep networks.

![ResNet Process 6](img/ResNet%20Process%206.png)


### Code

```python
def forward(self, x: Tensor) -> Tensor: 
    identity = x 
    out = self.conv1(x) 
    out = self.bn1(out) 
    out = self.relu(out) 
    out = self.conv2(out) 
    out = self.bn2(out) 
    
    if self.downsample is not None: 
        identity = self.downsample(x) 
    
    out += identity 
    out = self.relu(out) 
    
    return out
```


Resnet18, 34, 50, 101, 152. These are pretrained models and the number indicates the number of layers in the architecture.

### Architecture
ResNet models introduce the concept of residual learning, where the network learns residual functions with reference to the layer inputs, rather than learning unreferenced functions. This allows the model to train very deep networks effectively.

### Use Cases
ResNet models are commonly used in image classification tasks and are known for their performance on large-scale datasets such as ImageNet.

### Strengths and Weaknesses
- **Strengths**: 
  - Effective in training very deep networks.
  - Reduces the problem of vanishing gradients.
- **Weaknesses**: 
  - Computationally expensive for very deep versions like ResNet-152.
  - May not be as efficient for smaller datasets or less complex tasks.

### Papers
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778*.