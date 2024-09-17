import torch
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Set device to MPS for MacOS with MPS support
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Part Q1(a) - Use the same weight vector w* from Q1(a)
w0_star = torch.FloatTensor(1).uniform_(-1/4, 1/4).to(device)
w1_star = torch.FloatTensor(1).uniform_(-1, 1).to(device)
w2_star = torch.FloatTensor(1).uniform_(-1, 1).to(device)
w_star = torch.tensor([w0_star.item(), w1_star.item(), w2_star.item()], device=device)
print("Sampled weight vector w*:", w_star.cpu().numpy())

### Q2(d) - Repeat Q2(b) 100 Times with Random Initialization

n = 100
x1 = torch.FloatTensor(n).uniform_(-1, 1).to(device)
x2 = torch.FloatTensor(n).uniform_(-1, 1).to(device)
x0 = torch.ones(n).to(device)
X = torch.stack((x0, x1, x2), dim=1).to(device)

# Compute labels y = step(w*^T x) for the data set
y = (X @ w_star > 0).float() * 2 - 1  # Convert y to +1 or -1

# Function to run the Perceptron learning algorithm for different initial weights
def run_perceptron(X, y, eta, max_epochs=100):
    """Run Perceptron Learning Algorithm with given learning rate eta and random initial weights."""
    w_init = torch.FloatTensor(3).uniform_(-1, 1).to(device)  # Random initial weights
    errors = []

    for epoch in range(max_epochs):
        total_errors = 0
        for i in range(len(y)):
            # Calculate the predicted label y_w(x) = sign(w^T x)
            y_pred = torch.sign(w_init @ X[i])
            
            # Check if the predicted label is not equal to the true label
            if y_pred != y[i]:
                total_errors += 1
                # Update weights when there's a misclassification
                w_init += eta * y[i] * X[i]
        
        errors.append(total_errors)
        if total_errors == 0:
            break
    return errors

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
    # Convert errors to NumPy array after moving to CPU
    errors_matrix = np.array([np.pad(np.array(err, dtype=np.float32), (0, max_epochs - len(err)), 'constant', constant_values=np.nan) for err in all_results[eta]])
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

