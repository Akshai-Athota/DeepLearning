# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


image = torch.tensor(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=torch.float32,
)


image = image.unsqueeze(0).unsqueeze(0)

print(image.shape)

# %%
kernel = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)


kernel = kernel.unsqueeze(0).unsqueeze(0)
# %%
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

with torch.no_grad():
    conv.weight = nn.Parameter(kernel)

# %%
output = conv(image)
print(output)
# %%
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image[0, 0], cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output[0, 0].detach(), cmap="gray")
plt.title("After Convolution (Edges)")
plt.axis("off")

plt.show()

# %%
