import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Check for MPS support
device = torch.device("mps") if torch.has_mps else torch.device("cpu")
print(f"Using device: {device}")

# Load the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the convolutional neural network architecture with batch normalization
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm2d(20)  # Batch normalization after first conv layer
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(20)  # Batch normalization after second conv layer
        self.dropout2 = nn.Dropout(p=0.3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(20 * 5 * 5, 250)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))  # Apply batch norm and ReLU
        x = self.dropout1(x)

        x = torch.relu(self.bn2(self.conv2(x)))  # Apply batch norm and ReLU
        x = self.dropout2(x)

        x = self.pool(x)
        x = x.view(-1, 20 * 5 * 5)  # Flatten the channels into a vector
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


# Create the model, loss function, and optimizer with weight decay
model = ConvolutionalNeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=0.002, weight_decay=0.0001
)  # Higher learning rate

# Train the network
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%"
    )

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Report the chosen hyperparameters
print("Final Hyperparameters:")
print(f"Learning Rate: 0.002")  # Increased learning rate
print(f"Batch Size: 64")
print(f"Number of Epochs: {num_epochs}")
print(f"Weight Decay: 0.0001")
print(f"Dropout after conv1: 30%, conv2: 30%, fc1: 50%")
print(f"Batch Normalization after conv1 and conv2")
