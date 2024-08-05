import torch
import torch.nn as nn

# Define a simple model
model = nn.Linear(10, 2)
model.eval()  # Set the model to evaluation mode

# Dummy input tensor
input_tensor = torch.randn(1, 10)

# Perform inference without tracking gradients
with torch.no_grad():
    output = model(input_tensor)

# Check if gradients are being tracked
print(output.requires_grad)  # False
