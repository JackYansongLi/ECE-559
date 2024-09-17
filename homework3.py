import torch
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Set device to MPS for MacOS with MPS support
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Part Q1(a) - Sample random weights
w0_star = torch.FloatTensor(1).uniform_(-1/4, 1/4).to(device)
w1_star = torch.FloatTensor(1).uniform_(-1, 1).to(device)
w2_star = torch.FloatTensor(1).uniform_(-1, 1).to(device)
w_star = torch.tensor([w0_star.item(), w1_star.item(), w2_star.item()], device=device)
print("Sampled weight vector w*:", w_star.cpu().numpy())

# Part Q1(b) and (c) - Generate data and plot
n = 100
x1 = torch.FloatTensor(n).uniform_(-1, 1).to(device)
x2 = torch.FloatTensor(n).uniform_(-1, 1).to(device)
x0 = torch.ones(n).to(device)
X = torch.stack((x0, x1, x2), dim=1).to(device)

# Compute labels y = step(w*^T x)
y = (X @ w_star > 0).float() * 2 - 1  # Convert y to +1 or -1

# Scatter plot for data points
plt.scatter(x1[y == 1].cpu().numpy(), x2[y == 1].cpu().numpy(), color='red', label='y = 1')
plt.scatter(x1[y == -1].cpu().numpy(), x2[y == -1].cpu().numpy(), color='blue', label='y = -1')

# Plot the decision boundary w*^T x = 0
x_line = torch.linspace(-1, 1, 100)
y_line = -(w0_star.item() + w1_star.item() * x_line) / w2_star.item()
plt.plot(x_line.cpu().numpy(), y_line.cpu().numpy(), color='black', linestyle='--', label='w*^T x = 0')

# Plot the normal vector [w1*, w2*] starting from the origin (0, 0)
plt.arrow(0, 0, w1_star.item(), w2_star.item(), head_width=0.1, head_length=0.1, fc='green', ec='green', label='Normal vector')

# Set equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.legend()
plt.title('Scatter Plot of Linearly Separable Classes with Normal Vector')
plt.grid(True)
plt.show()

### Q2(a) - Implement Perceptron Learning Algorithm for η = 1

def run_perceptron(X, y, eta, max_epochs=100):
    """Run Perceptron Learning Algorithm with given learning rate eta."""
    w = torch.tensor([1.0, 1.0, 1.0], device=device, requires_grad=False)  # Initial weights
    errors = []

    for epoch in range(max_epochs):
        total_errors = 0
        for i in range(n):
            # Calculate the predicted label y_w(x) = sign(w^T x)
            y_pred = torch.sign(w @ X[i])
            
            # Check if the predicted label is not equal to the true label
            if y_pred != y[i]:
                total_errors += 1
                # Update weights when there's a misclassification
                w += eta * y[i] * X[i]
        
        errors.append(total_errors)
        if total_errors == 0:
            break
    return w, errors

# Run Perceptron for η = 1
eta_1 = 1
w_1, errors_1 = run_perceptron(X, y, eta_1)

# Plot the decision boundary from Perceptron for η = 1
y_line_perceptron_1 = -(w_1[0].item() + w_1[1].item() * x_line) / w_1[2].item()

plt.scatter(x1[y == 1].cpu().numpy(), x2[y == 1].cpu().numpy(), color='red', label='y = 1')
plt.scatter(x1[y == -1].cpu().numpy(), x2[y == -1].cpu().numpy(), color='blue', label='y = -1')

# Original decision boundary
plt.plot(x_line.cpu().numpy(), y_line.cpu().numpy(), color='black', linestyle='--', label='w*^T x = 0')
# Decision boundary from Perceptron with η = 1
plt.plot(x_line.cpu().numpy(), y_line_perceptron_1.cpu().numpy(), color='green', linestyle='-', label='Perceptron boundary (η = 1)')

plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.legend()
plt.title('Perceptron Learning Algorithm with Decision Boundary (η = 1)')
plt.grid(True)
plt.show()

# Report the final weights and errors for η = 1
print("Final weights from Perceptron for η = 1:", w_1.cpu().numpy())
print("Errors in each epoch for η = 1:", errors_1)
### Q2(b) - Run Perceptron for Different Learning Rates and Plot Errors

# Learning rates to evaluate
etas = [1, 0.1, 10]
colors = ['green', 'orange', 'purple']
results = {}

# Run Perceptron for each eta and store results
for eta, color in zip(etas, colors):
    w, errors = run_perceptron(X, y, eta)
    results[eta] = (w, errors)
    plt.plot(range(1, len(errors) + 1), errors, color=color, label=f'η = {eta}')

# Plot the error convergence for different learning rates
plt.xlabel('Epoch')
plt.ylabel('Number of Errors')
plt.title('Perceptron Learning Algorithm: Errors vs. Epoch for Different η')
plt.legend()
plt.grid(True)
plt.show()

# Report the final weights and errors for each η
for eta in etas:
    w, errors = results[eta]
    print(f"Final weights for η = {eta}: {w.cpu().numpy()}")
    print(f"Errors per epoch for η = {eta}: {errors}")
### Q2(c) - Generate Larger Data Set with n = 1000

n_large = 1000
x1_large = torch.FloatTensor(n_large).uniform_(-1, 1).to(device)
x2_large = torch.FloatTensor(n_large).uniform_(-1, 1).to(device)
x0_large = torch.ones(n_large).to(device)
X_large = torch.stack((x0_large, x1_large, x2_large), dim=1).to(device)

# Compute labels y = step(w*^T x) for the larger data set
y_large = (X_large @ w_star > 0).float() * 2 - 1  # Convert y to +1 or -1
# Run Perceptron for η = 1 with larger dataset
eta_1 = 1
w_large, errors_large = run_perceptron(X_large, y_large, eta_1)

# Plot the decision boundary from Perceptron for η = 1 with larger dataset
y_line_perceptron_large = -(w_large[0].item() + w_large[1].item() * x_line) / w_large[2].item()

plt.scatter(x1_large[y_large == 1].cpu().numpy(), x2_large[y_large == 1].cpu().numpy(), color='red', label='y = 1')
plt.scatter(x1_large[y_large == -1].cpu().numpy(), x2_large[y_large == -1].cpu().numpy(), color='blue', label='y = -1')

# Original decision boundary
plt.plot(x_line.cpu().numpy(), y_line.cpu().numpy(), color='black', linestyle='--', label='w*^T x = 0')
# Decision boundary from Perceptron with η = 1 on larger dataset
plt.plot(x_line.cpu().numpy(), y_line_perceptron_large.cpu().numpy(), color='green', linestyle='-', label='Perceptron boundary (η = 1, n = 1000)')

plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.legend()
plt.title('Perceptron Learning Algorithm with Larger Data Set (η = 1, n = 1000)')
plt.grid(True)
plt.show()

# Report the final weights and errors for η = 1 with larger dataset
print("Final weights from Perceptron for η = 1 with larger data set:", w_large.cpu().numpy())
print("Errors in each epoch for η = 1 with larger data set:", errors_large)

# Compare the final weights w and w*
print("Original weight vector w*:", w_star.cpu().numpy())
print("Difference between w* and w obtained with larger dataset:", (w_large - w_star).cpu().numpy())
# Initialize variables
etas = [0.1, 1, 10]
colors = ['green', 'orange', 'purple']
num_repetitions = 100
max_epochs = 100

# Prepare to store results
all_results = {eta: [] for eta in etas}

# Run Perceptron for each eta with 100 repetitions
for eta in etas:
    for _ in range(num_repetitions):
        errors = run_perceptron(X, y, eta)
        all_results[eta].append(errors)

# Calculate average, 10th percentile, and 90th percentile
for eta, color in zip(etas, colors):
    errors_matrix = np.array([np.pad(err, (0, max_epochs - len(err)), 'constant', constant_values=np.nan) for err in all_results[eta]])
    mean_errors = np.nanmean(errors_matrix, axis=0)
    percentile_10 = np.nanpercentile(errors_matrix, 10, axis=0)
    percentile_90 = np.nanpercentile(errors_matrix, 90, axis=0)

    # Plot the results
    plt.plot(range(1, max_epochs + 1), mean_errors, color=color, label=f'η = {eta}')
    plt.fill_between(range(1, max_epochs + 1), percentile_10, percentile_90, color=color, alpha=0.3)

plt.xlabel('Epoch')
plt.ylabel('Average Number of Errors')
plt.title('Perceptron Learning Algorithm: Errors vs. Epoch for Different η with Random Initialization')
plt.legend()
plt.grid(True)
plt.show()
