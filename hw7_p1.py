import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set device to use MPS backend
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

# Load MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the neural network
class ClusteringModel(nn.Module):
    def __init__(self, num_centers=10, input_dim=784):
        super(ClusteringModel, self).__init__()
        self.centers = nn.Parameter(
            torch.randn(num_centers, input_dim)
        )  # 10 x 784 centers
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Flatten the input
        x0 = self.flatten(x)

        # Calculate the distance-based activation
        x = (
            torch.matmul(x0, self.centers.t())
            - 0.5 * torch.sum(self.centers**2, dim=1).flatten()
        )
        x = 20 * x
        x = self.softmax(x)

        # Reconstruct the input using the center
        x_reconstructed = torch.matmul(x, self.centers)

        # Return the reconstruction error
        return x_reconstructed, x0


# Instantiate the model
model = ClusteringModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        x_reconstructed, x0 = model(data)
        loss = criterion(x_reconstructed, x0)

        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Train Loss: {total_loss / len(train_loader):.4f}")


# Testing loop
def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            # Forward pass
            x_reconstructed, x0 = model(data)
            loss = criterion(x_reconstructed, x0)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(test_loader):.4f}")


# # Training and evaluation
# num_epochs = 10
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}/{num_epochs}")
#     train_model(model, train_loader, optimizer, criterion, device)
#     test_model(model, test_loader, criterion, device)


# Initialize cluster centers to specific digits
def initialize_cluster_centers_by_digits(model, train_loader, device):
    model.centers.data.zero_()  # Reset centers to zeros
    digit_count = [0] * 10
    centers_initialized = torch.zeros(10, 784).to(device)

    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)
        for img, label in zip(data, labels):
            digit = label.item()
            if digit_count[digit] == 0:  # Only take the first image of each digit
                centers_initialized[digit] = img.view(-1)
                digit_count[digit] += 1

        if sum(digit_count) == 10:  # Stop once we have initialized all 10 digits
            break

    model.centers.data = centers_initialized.clone()


# Generate a confusion matrix
def generate_confusion_matrix(model, train_loader, device):
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int32)

    with torch.no_grad():
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            x_reconstructed, x0 = model(data)
            softmax_output = F.softmax(torch.matmul(x0, model.centers.t()), dim=1)
            predicted_centers = torch.argmax(softmax_output, dim=1)

            for label, prediction in zip(labels, predicted_centers):
                confusion_matrix[label.item(), prediction.item()] += 1

    return confusion_matrix


# Main script for part (b)
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

# Prepare data loader with batch size 1
batch_size_1_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# # Initialize cluster centers with specific digits
# model = ClusteringModel().to(device)
# initialize_cluster_centers_by_digits(model, batch_size_1_loader, device)

# # Train the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}/{num_epochs}")
#     train_model(model, train_loader, optimizer, criterion, device)

# # Collect final cluster centers
# final_centers = model.centers.cpu().detach().numpy()

# # Generate the confusion matrix
# confusion_matrix = generate_confusion_matrix(model, train_loader, device)

# # Print outputs
# print("Final Cluster Centers:")
# print(final_centers)

# print("\nConfusion Matrix (Rows: True Digits, Columns: Cluster Assignments):")
# print(confusion_matrix.numpy())


# Initialize cluster centers uniformly from [0, 1]
def initialize_centers_uniform_random(model, device):
    model.centers.data = torch.rand_like(model.centers).to(device)


# Main script for random initialization
model = ClusteringModel().to(device)
initialize_centers_uniform_random(model, device)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_model(model, train_loader, optimizer, criterion, device)

# Collect final cluster centers
final_centers_uniform = model.centers.cpu().detach().numpy()

# Generate the confusion matrix
confusion_matrix_uniform = generate_confusion_matrix(model, train_loader, device)

# Print outputs
print("Final Cluster Centers (Uniform Initialization):")
print(final_centers_uniform)

print("\nConfusion Matrix (Rows: True Digits, Columns: Cluster Assignments):")
print(confusion_matrix_uniform.numpy())
