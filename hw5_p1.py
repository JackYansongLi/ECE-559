import torch
import matplotlib.pyplot as plt

# Set device to MPS (if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Function definition R(w)
def R(w1, w2):
    return 13 * w1**2 - 10 * w1 * w2 + 4 * w1 + 2 * w2**2 - 2 * w2 + 1

# Gradient of R(w)
def gradient(w):
    w1, w2 = w[0], w[1]
    grad_w1 = 26 * w1 - 10 * w2 + 4
    grad_w2 = -10 * w1 + 4 * w2 - 2
    return torch.tensor([grad_w1, grad_w2], device=device)

# Optimal solution for R(w)
optimal_solution = torch.tensor([1.0, 3.0], device=device)


# Perform gradient descent
def gradient_descent(eta, max_iters=500):
    # Initial guess for w1 and w2
    w = torch.tensor([0.0, 0.0], device=device, requires_grad=False)
    
    # Store distances at each iteration
    distances = []
    
    for i in range(max_iters):
        # Compute gradient
        grad = gradient(w)
        
        # Update w using the learning rate eta
        w = w - eta * grad
        
        # Compute the distance to the optimal solution
        distance = torch.norm(w - optimal_solution).item()
        distances.append(distance)
    
    return distances

# Plot distances for each learning rate
etas = [0.02, 0.05, 0.1]
max_iters = 500

plt.figure(figsize=(10, 6))

for eta in etas:
    distances = gradient_descent(eta, max_iters)
    plt.plot(distances, label=f'eta={eta}')

plt.xlabel('Iteration')
plt.ylabel('Distance to Optimal Solution (Log Scale)')
plt.yscale('log')  # Set y-axis to log scale
plt.title('Log Scale Distance vs. Iteration for Different Learning Rates')
plt.legend()
plt.grid(True)
plt.show()
