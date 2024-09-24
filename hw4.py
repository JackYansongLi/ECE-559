import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv

# Check if MPS (Apple Silicon GPU) is available, else fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Download MNIST dataset
transform = torchvision.transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Combine train and test datasets
dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

# Extract the raw data (Xraw and Yraw)
Xraw = torch.stack([img[0] for img in dataset], dim=0).reshape(-1, 28*28).to(device)
Yraw = torch.tensor([img[1] for img in dataset], dtype=torch.long).to(device)

# Print shape of Xraw and Yraw
print(f"Xraw shape: {Xraw.shape}")  # Expected shape: [70000, 784]
print(f"Yraw shape: {Yraw.shape}")  # Expected shape: [70000]

# Plot one example of each digit (0-9)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()  # Flatten the axes for easy iteration
for digit in range(10):
    # Find the first occurrence of each digit in the dataset
    idx = (Yraw == digit).nonzero(as_tuple=True)[0][0].item()
    
    # Reshape and plot the digit
    img = Xraw[idx].reshape(28, 28).cpu()
    axes[digit].imshow(img, cmap='gray')
    axes[digit].set_title(f"Digit: {digit}")
    axes[digit].axis('off')

plt.tight_layout()
plt.show()
# Function to create matrix M and transformed X
def create_transformed_X(Xraw, d):
    M = torch.rand(d, 784).to(device) / (255 * d)
    X_transposed = Xraw.T
    X = M @ X_transposed
    return X

# Function to create one-hot encoded Y
def create_one_hot_Y(Yraw):
    Y = torch.zeros(10, Yraw.size(0)).to(device)
    Y.scatter_(0, Yraw.unsqueeze(0), 1)
    return Y

# Function to calculate MSE and the number of mistakes
def calculate_mse_and_mistakes(X, Y):
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    # Calculate pseudoinverse and best weights W
    X_pinv = pinv(X_np)
    W = Y_np @ X_pinv
    
    # Predict Y using f(x; W) = W @ X
    Y_pred = W @ X_np
    
    # Calculate MSE
    mse = np.mean(np.square(Y_np - Y_pred))
    
    # Calculate mistakes (argmax comparison)
    pred_labels = np.argmax(Y_pred, axis=0)
    true_labels = np.argmax(Y_np, axis=0)
    mistakes = np.sum(pred_labels != true_labels)
    
    return mse, mistakes

# Set values of d and run experiments
d_values = [10, 50, 100, 200, 500]

# Store results for each d
results = []

for d in d_values:
    # Step 1: Transform X for the current d
    X = create_transformed_X(Xraw, d)
    
    # Step 2: Create one-hot encoded Y
    Y = create_one_hot_Y(Yraw)
    
    # Step 3: Calculate MSE and mistakes
    mse, mistakes = calculate_mse_and_mistakes(X, Y)
    
    # Store the results
    results.append((d, mse, mistakes))
    print(f"d={d}, MSE={mse:.4f}, Mistakes={mistakes}")
# Implementing Widrow-Hoff LMS algorithm
def widrow_hoff_lms(X, Y, d, eta=0.001, epochs=10):
    # Initialize weights at the origin
    W = torch.zeros((10, d), device=device)
    mse_per_epoch = []
    
    # LMS loop over epochs
    for epoch in range(epochs):
        for i in range(X.shape[1]):
            xi = X[:, i].unsqueeze(1)  # Get the i-th column (one input sample)
            yi = Y[:, i].unsqueeze(1)  # Get the corresponding one-hot encoded label
            
            # Prediction: f(xi, W) = W @ xi
            y_pred = W @ xi
            
            # Update the weight using Widrow-Hoff rule
            W += eta * (yi - y_pred) @ xi.T
        
        # Compute the MSE at the end of the epoch
        Y_pred = W @ X
        mse = torch.mean((Y - Y_pred) ** 2).item()
        mse_per_epoch.append(mse)
        print(f"Epoch {epoch+1}/{epochs}, MSE: {mse:.6f}")
    
    return W, mse_per_epoch

# Function to calculate the number of mistakes
def calculate_mistakes(W, X, Y):
    # Predict Y using f(x; W) = W @ X
    Y_pred = W @ X
    
    # Argmax to get predicted and true labels
    pred_labels = torch.argmax(Y_pred, axis=0)
    true_labels = torch.argmax(Y, axis=0)
    
    # Count mistakes
    mistakes = torch.sum(pred_labels != true_labels).item()
    return mistakes
# Set d = 100 and run Widrow-Hoff LMS algorithm
d = 100
eta = 0.001
epochs = 10

# Step 1: Transform X for the current d
X = create_transformed_X(Xraw, d)

# Step 2: Create one-hot encoded Y
Y = create_one_hot_Y(Yraw)

# Step 3: Run the Widrow-Hoff LMS algorithm
W_lms, mse_per_epoch = widrow_hoff_lms(X, Y, d, eta=eta, epochs=epochs)

# Step 4: Plot the MSE vs. the number of epochs
plt.plot(range(1, epochs + 1), mse_per_epoch, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE vs. Number of Epochs (LMS Algorithm)')
plt.grid(True)
plt.show()

# Step 5: Calculate the number of mistakes
mistakes = calculate_mistakes(W_lms, X, Y)
print(f"Number of mistakes after training: {mistakes}")
