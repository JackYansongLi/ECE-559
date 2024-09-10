import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Define the neural network model
class LogicNet(nn.Module):
    def __init__(self):
        super(LogicNet, self).__init__()
        # Define the weights and biases according to the corrected logic
        self.hidden_weights = torch.tensor(
            [[1.0, 1.0, -1.0], [0.0, -1.0, 1.0]], device=device
        )
        self.hidden_bias = torch.tensor([-2, -1], device=device)
        self.output_weights = torch.tensor([1.0, 1.0], device=device)
        self.output_bias = torch.tensor([1], device=device)

    def forward(self, x):
        # Hidden layer computation
        h = torch.sign(torch.matmul(x, self.hidden_weights.T) + self.hidden_bias)
        # Output layer computation
        y = torch.sign(torch.matmul(h, self.output_weights) + self.output_bias)
        return y


# Instantiate the model
model = LogicNet().to(device)

# Define the input table (truth table for 3 variables)
inputs = torch.tensor(
    [
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ],
    dtype=torch.float,
).to(device)


# Function to convert numerical inputs to boolean
def to_bool(x):
    return x == 1


# Generate expected outputs using Python's logical operators
expected_outputs = torch.tensor(
    [
        [
            (
                1
                if (to_bool(x1) and to_bool(x2) and not to_bool(x3))
                or (not to_bool(x2) and to_bool(x3))
                else -1
            )
        ]
        for x1, x2, x3 in inputs.tolist()
    ],
    dtype=torch.float,
).to(device)

# # Print the expected outputs
# print("Expected Outputs:")
# for i, output in enumerate(expected_outputs):
#     print(f"Input: {inputs[i].tolist()} => Expected Output: {output.item()}")

# Compute the output using the neural network
with torch.no_grad():
    output = model(inputs)
    print("\nInput-Output Table:")
    print(f"{'Inputs':<20}{'Output':<10}{'Expected':<10}")
    for i in range(len(inputs)):
        print(
            f"{str(inputs[i].tolist()):<20}{output[i].item():<10}{expected_outputs[i].item():<10}"
        )

# # Verify the results
# matches = (output == expected_outputs).all().item()
# print(f"\nVerification: {'Matched' if matches else 'Did Not Match'}")


# Verify the outputs match expected
# Testing the model
with torch.no_grad():
    test_outputs = model(inputs)
# Reshape the test_outputs to match the shape of expected_outputs
test_outputs_reshaped = test_outputs.view(-1, 1)

# Verify the outputs match expected
verification_result = torch.equal(test_outputs_reshaped, expected_outputs)
print("Verification: ", verification_result)


# Define the neural network model based on the given weights and biases
class CustomNN(nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        # Define parameters based on given matrices and vectors
        self.W = torch.tensor([[1.0, -1.0], [-1.0, -1.0], [0.0, -1.0]], device=device)
        self.b = torch.tensor([1.0, 1.0, -1.0], device=device)
        self.u = torch.tensor([1.0, 1.0, -1.0], device=device)
        self.c = -1.5

    def forward(self, x):
        # Step activation function
        step = lambda v: (v >= 0).float()

        # Compute hidden layer
        h = step(torch.matmul(self.W, x.T) + self.b.view(-1, 1))

        # Compute output layer
        y = step(torch.matmul(self.u, h) + self.c)

        return y


# Instantiate the model
model = CustomNN().to(device)

# Generate 1000 random points from a uniform distribution over [-2, 2]^2
num_points = 1000
points = torch.FloatTensor(num_points, 2).uniform_(-2, 2).to(device)

# Compute outputs for all points
outputs = model(points)

# Convert outputs to CPU for plotting
points_cpu = points.cpu().numpy()
outputs_cpu = outputs.cpu().numpy()

# Plot the scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(
    points_cpu[:, 0],
    points_cpu[:, 1],
    c=["blue" if y == 0 else "red" for y in outputs_cpu],
    marker=".",
)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Scatter Plot of Neural Network Outputs")
plt.grid(True)

# Save the plot as a vector graphic
plt.savefig("./decision_boundary.jpg", format="jpg")

plt.show()
