import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

a = 5  # Parameter for the sigmoid


# Sigmoid function using PyTorch
def sigmoid(z):
    return 1 / (1 + torch.exp(-a * z))


# Derivative of sigmoid function
def sigmoid_derivative(z):
    sig = sigmoid(z)
    return a * sig * (1 - sig)


# Define the squared loss function
def squared_loss(y, f):
    return (f - y) ** 2


# Neural Network model using sigmoid activation
class CustomNN(nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        # Define parameters based on given matrices and vectors
        self.W = torch.tensor(
            [[1.0, -1.0], [-1.0, -1.0], [0.0, -1.0]], dtype=torch.float32
        )
        self.b = torch.tensor([[1.0], [1.0], [-1.0]], dtype=torch.float32)
        self.U = torch.tensor([[1.0, 1.0, -1.0]], dtype=torch.float32)
        self.c = torch.tensor([[-1.5]], dtype=torch.float32)

    def forward(self, x):
        # Compute the hidden layer (v_z)
        v_z = torch.matmul(self.W, x.T) + self.b

        # Apply sigmoid activation to hidden layer output (z)
        z = sigmoid(v_z)

        # Compute the output layer (v_f)
        v_f = torch.matmul(self.U, z) + self.c

        # Apply sigmoid activation to final output (f)
        y = sigmoid(v_f)

        return y, z


# Neural Network model using sigmoid activation
class TrainingNN(nn.Module):
    def __init__(self):
        super(TrainingNN, self).__init__()
        # Initialize W, b, U, and c with random values of the same size
        self.W = torch.randn(3, 2, dtype=torch.float32)  # Random values for W
        self.b = torch.randn(3, 1, dtype=torch.float32)  # Random values for b
        self.U = torch.randn(1, 3, dtype=torch.float32)  # Random values for U
        self.c = torch.randn(1, 1, dtype=torch.float32)  # Random value for c

    def forward(self, x):
        # Compute the hidden layer (v_z)
        v_z = torch.matmul(self.W, x.T) + self.b

        # Apply sigmoid activation to hidden layer output (z)
        z = sigmoid(v_z)

        # Compute the output layer (v_f)
        v_f = torch.matmul(self.U, z) + self.c

        # Apply sigmoid activation to final output (f)
        f = sigmoid(v_f)

        return f, z  # Return f and z (for use in the backward pass)


# Define the backward pass
def backward_pass(x, y, model):
    # Forward pass to get f and intermediate values
    v_z = torch.matmul(model.W, x.T) + model.b
    z = sigmoid(v_z)
    v_f = torch.matmul(model.U, z) + model.c
    f = sigmoid(v_f)

    # Compute loss
    loss = squared_loss(y, f)

    # Compute delta_f (gradient of loss w.r.t. f) - Ensure it's for a single sample
    delta_f = 2 * (f - y)  # This should result in a torch.Size([1, 1]) for each sample

    # Compute gradient w.r.t. v_f
    delta_v_f = delta_f * sigmoid_derivative(v_f)  # Should be torch.Size([1, 1])

    # Compute gradients for U and c
    grad_U = torch.matmul(delta_v_f, z.T)  # grad_U should be torch.Size([1, 3])
    grad_c = delta_v_f  # grad_c should be torch.Size([1, 1])

    # Backpropagate through the second layer to z
    delta_z = torch.matmul(model.U.T, delta_v_f)  # delta_z should be torch.Size([3, 1])

    # Compute gradient w.r.t. v_z
    delta_v_z = delta_z * sigmoid_derivative(
        v_z
    )  # delta_v_z should be torch.Size([3, 1])

    # Compute gradients for W and b
    grad_W = torch.matmul(delta_v_z, x)  # grad_W should be torch.Size([3, 2])
    grad_b = delta_v_z  # grad_b should be torch.Size([3, 1])

    return grad_W, grad_b, grad_U, grad_c, loss


# Instantiate the model
model = TrainingNN()
real_model = CustomNN()

# Generate 1000 random points from a uniform distribution over [-2, 2]^2 using PyTorch tensors
num_points = 1000
points = torch.FloatTensor(num_points, 2).uniform_(-2, 2)  # Inputs
outputs = real_model.forward(points)[
    0
].detach()  # Outputs (used as target labels for this example)

# Store the targets (Y) as outputs from the model (since no ground truth is given)
Y = outputs

# Define learning rate and number of epochs
learning_rate = 0.01
num_epochs = 10

# Placeholder for storing MSE for each epoch
mse_per_epoch = []

# Training loop (Stochastic Gradient Descent)
for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(num_points):
        x = points[i].unsqueeze(0)  # Single data point (1x2)
        y = Y[:, i]  # Corresponding target (1x1)

        # Perform backward pass on single point
        grad_W, grad_b, grad_U, grad_c, loss = backward_pass(x, y, model)

        # Update weights using stochastic gradient descent (SGD)
        model.W -= learning_rate * grad_W
        model.b -= learning_rate * grad_b
        model.U -= learning_rate * grad_U
        model.c -= learning_rate * grad_c

        # Accumulate loss for MSE calculation
        total_loss += loss.item()

    # Compute mean squared error for this epoch and store it
    mse = total_loss / num_points
    mse_per_epoch.append(mse)
    print(f"Epoch {epoch+1}/{num_epochs}, MSE: {mse:.4f}")

# Plotting MSE vs. Epoch
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), mse_per_epoch, marker="o", color="b")
plt.title("MSE vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.savefig("mse_vs_epoch.png", format="png")
plt.show()

# Create a grid of points in the range [-2, 2] for both x1 and x2
x1_range = np.linspace(-2, 2, 50)
x2_range = np.linspace(-2, 2, 50)
x1, x2 = np.meshgrid(x1_range, x2_range)

# Flatten the grid so we can input it into the model
x1_flat = x1.flatten()
x2_flat = x2.flatten()

# Stack x1 and x2 into a tensor for the model input
inputs = torch.tensor(np.vstack([x1_flat, x2_flat]).T, dtype=torch.float32)

# Pass the grid points through the trained model to get the predictions
with torch.no_grad():  # Disable gradient computation for evaluation
    predictions, _ = model.forward(inputs)

# Convert continuous predictions to binary (0 or 1)
y_pred_binary = (predictions >= 0.5).float().numpy().flatten()

# Separate points by class for plotting
class_0 = inputs[y_pred_binary == 0].numpy()
class_1 = inputs[y_pred_binary == 1].numpy()

# Create a 3D scatter plot
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

# Scatter plot for class 0 (blue)
ax.scatter(
    class_0[:, 0],
    class_0[:, 1],
    np.zeros_like(class_0[:, 0]),
    color="blue",
    label="y=0",
)

# Scatter plot for class 1 (red)
ax.scatter(
    class_1[:, 0], class_1[:, 1], np.ones_like(class_1[:, 0]), color="red", label="y=1"
)

# Labels and legend
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.legend()

plt.title("3D Decision Boundary Visualization")
plt.show()
