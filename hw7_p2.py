# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

# # Set device to use GPU if available, else CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# # Load MNIST dataset
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = datasets.MNIST(
#     root="./data", train=True, download=True, transform=transform
# )
# test_dataset = datasets.MNIST(
#     root="./data", train=False, download=True, transform=transform
# )
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# # Define the Autoencoder
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()

#         # Encoder
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=1)
#         self.bn1 = nn.BatchNorm2d(20)
#         self.dropout1 = nn.Dropout(0.1)

#         self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=4, stride=2)
#         self.bn2 = nn.BatchNorm2d(20)
#         self.dropout2 = nn.Dropout(0.1)

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.fc1 = nn.Linear(20 * 5 * 5, 250)
#         self.bn3 = nn.BatchNorm1d(250)
#         self.dropout3 = nn.Dropout(0.1)

#         self.fc2 = nn.Linear(250, 10)
#         # No batchnorm or dropout after last layer

#         # Decoder
#         self.fc3 = nn.Linear(10, 360)
#         self.bn4 = nn.BatchNorm1d(360)
#         self.dropout4 = nn.Dropout(0.1)

#         self.fc4 = nn.Linear(360, 720)
#         self.bn5 = nn.BatchNorm1d(720)
#         self.dropout5 = nn.Dropout(0.1)

#         # ConvTranspose layers with adjusted parameters
#         self.convTrans1 = nn.ConvTranspose2d(
#             in_channels=20,
#             out_channels=20,
#             kernel_size=5,
#             stride=2,
#             padding=0,
#             output_padding=1,
#         )
#         self.bn6 = nn.BatchNorm2d(20)
#         self.dropout6 = nn.Dropout(0.1)

#         self.convTrans2 = nn.ConvTranspose2d(
#             in_channels=20, out_channels=1, kernel_size=3, stride=1, padding=1
#         )
#         # Followed by sigmoid activation

#     def forward(self, x, enc_mode=1):
#         if enc_mode == 1:
#             x_orig = x.clone()
#             # Encoder
#             x = F.relu(self.conv1(x))
#             x = self.bn1(x)
#             x = self.dropout1(x)

#             x = F.relu(self.conv2(x))
#             x = self.bn2(x)
#             x = self.dropout2(x)

#             x = self.pool(x)

#             x = x.view(-1, 20 * 5 * 5)

#             x = F.relu(self.fc1(x))
#             x = self.bn3(x)
#             x = self.dropout3(x)

#             z = self.fc2(x)  # z has shape (batch_size, 10)

#             # Add reduced noise to z
#             z2 = z + torch.randn(z.shape).to(z.device) * 0.1  # Reduced noise
#         else:
#             batch_size = x.size(0)
#             x_orig = torch.zeros_like(x)
#             z = torch.zeros(batch_size, 10).to(x.device)  # Placeholder, not used
#             z2 = torch.randn(batch_size, 10).to(x.device)

#         # Decoder
#         x = F.relu(self.fc3(z2))
#         x = self.bn4(x)
#         x = self.dropout4(x)

#         x = F.relu(self.fc4(x))
#         x = self.bn5(x)
#         x = self.dropout5(x)

#         x = x.view(-1, 20, 6, 6)  # Unflatten to (batch_size, 20, 6, 6)

#         x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

#         x = F.relu(self.convTrans1(x))
#         x = self.bn6(x)
#         x = self.dropout6(x)

#         x = self.convTrans2(x)
#         x = torch.sigmoid(x)

#         f = x  # Reconstructed image

#         # Compute reconstruction error
#         recon_error = (f - x_orig).view(f.size(0), -1)

#         # Concatenate z and reconstruction error along dimension 1
#         output = torch.cat((z, recon_error), dim=1)

#         return output, f  # Return reconstructed image for visualization


# # Instantiate the model
# model = Autoencoder().to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate


# # Training loop
# def train_model(model, train_loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()

#         # Forward pass
#         output, _ = model(data, enc_mode=1)
#         target = torch.zeros_like(output).to(device)
#         loss = criterion(output, target)

#         # Backward pass
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     print(f"Train Loss: {total_loss / len(train_loader):.4f}")


# # Testing loop
# def test_model(model, test_loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for data, _ in test_loader:
#             data = data.to(device)

#             # Forward pass
#             output, _ = model(data, enc_mode=1)
#             target = torch.zeros_like(output).to(device)
#             loss = criterion(output, target)
#             total_loss += loss.item()

#     print(f"Test Loss: {total_loss / len(test_loader):.4f}")


# # Training and evaluation
# num_epochs = 10  # Increased epochs
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}/{num_epochs}")
#     train_model(model, train_loader, optimizer, criterion, device)
#     test_model(model, test_loader, criterion, device)


# # Function to visualize reconstructed images
# def visualize_reconstructions(model, data_loader):
#     model.eval()
#     with torch.no_grad():
#         data_iter = iter(data_loader)
#         images, _ = next(data_iter)
#         images = images.to(device)
#         _, reconstructed = model(images, enc_mode=1)
#         images = images.cpu()
#         reconstructed = reconstructed.cpu()

#         # Plot original and reconstructed images
#         n = 10
#         plt.figure(figsize=(10, 4))
#         for i in range(n):
#             # Original images
#             ax = plt.subplot(2, n, i + 1)
#             plt.imshow(images[i].squeeze(), cmap="gray")
#             plt.axis("off")
#             # Reconstructed images
#             ax = plt.subplot(2, n, i + 1 + n)
#             plt.imshow(reconstructed[i].squeeze(), cmap="gray")
#             plt.axis("off")
#         plt.suptitle("Original Images (Top Row) and Reconstructed Images (Bottom Row)")
#         plt.show()


# # Visualize reconstructions on test set
# visualize_reconstructions(model, test_loader)


# # Generate and visualize 20 random digits
# def generate_and_visualize(model, num_images=20):
#     model.eval()
#     with torch.no_grad():
#         # Create dummy input (content doesn't matter since enc_mode=0)
#         dummy_input = torch.zeros(num_images, 1, 28, 28).to(device)
#         output, f = model(dummy_input, enc_mode=0)

#         # Generated images
#         generated_images = f.cpu()

#         # Plot the images
#         n = num_images
#         plt.figure(figsize=(10, 2))
#         for i in range(n):
#             ax = plt.subplot(2, n // 2, i + 1)
#             plt.imshow(generated_images[i].squeeze(), cmap="gray")
#             plt.axis("off")
#         plt.suptitle("Generated Images")
#         plt.tight_layout()
#         plt.show()


# # Call the function to generate and visualize images
# generate_and_visualize(model, num_images=20)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set device to use GPU if available, else MPS
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(
            1, 32, kernel_size=4, stride=2, padding=1
        )  # 14x14
        self.encoder_conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # 7x7
        self.encoder_fc1 = nn.Linear(64 * 7 * 7, 256)
        self.encoder_mu = nn.Linear(256, 10)  # Output: mean of z
        self.encoder_logvar = nn.Linear(256, 10)  # Output: log variance of z

        # Decoder
        self.decoder_fc1 = nn.Linear(10, 256)
        self.decoder_fc2 = nn.Linear(256, 64 * 7 * 7)
        self.decoder_convT1 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )
        self.decoder_convT2 = nn.ConvTranspose2d(
            32, 1, kernel_size=4, stride=2, padding=1
        )

    def encode(self, x):
        x = F.leaky_relu(self.encoder_conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.encoder_conv2(x), negative_slope=0.2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.leaky_relu(self.encoder_fc1(x), negative_slope=0.2)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar, enc_mode=1):
        if enc_mode == 1:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std  # Reparameterization trick
        else:
            # Generation mode: sample z from standard normal
            return torch.randn(mu.size(0), mu.size(1)).to(device)

    def decode(self, z):
        x = F.leaky_relu(self.decoder_fc1(z), negative_slope=0.2)
        x = F.leaky_relu(self.decoder_fc2(x), negative_slope=0.2)
        x = x.view(-1, 64, 7, 7)
        x = F.leaky_relu(self.decoder_convT1(x), negative_slope=0.2)
        x = torch.sigmoid(self.decoder_convT2(x))
        return x

    def forward(self, x, enc_mode=1):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, enc_mode)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# Instantiate the model
model = VAE().to(device)
criterion = nn.BCELoss(reduction="sum")  # Sum over all pixels
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=1e-5
)  # Added weight decay


# VAE loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = criterion(recon_x, x)
    # KLD divergence between the learned distribution and standard normal distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Return average loss per batch
    return (BCE + KLD) / x.size(0)


# Training loop
def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar = model(data, enc_mode=1)
        loss = loss_function(recon_batch, data, mu, logvar)

        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader.dataset)
    print(f"Train Loss: {average_loss:.4f}")


# Testing loop
def test_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            # Forward pass
            recon_batch, mu, logvar = model(data, enc_mode=1)
            loss = loss_function(recon_batch, data, mu, logvar)
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader.dataset)
    print(f"Test Loss: {average_loss:.4f}")


# Training and evaluation
num_epochs = 10  # Increased epochs
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_model(model, train_loader, optimizer, device)
    test_model(model, test_loader, device)


# Function to visualize reconstructed images
def visualize_reconstructions(model, data_loader):
    model.eval()
    with torch.no_grad():
        data_iter = iter(data_loader)
        images, _ = next(data_iter)
        images = images.to(device)
        recon_images, _, _ = model(images, enc_mode=1)
        images = images.cpu()
        recon_images = recon_images.cpu()

        # Plot original and reconstructed images
        n = 10
        plt.figure(figsize=(10, 4))
        for i in range(n):
            # Original images
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images[i].squeeze(), cmap="gray")
            plt.axis("off")
            # Reconstructed images
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon_images[i].squeeze(), cmap="gray")
            plt.axis("off")
        plt.suptitle("Original Images (Top Row) and Reconstructed Images (Bottom Row)")
        plt.show()


# Visualize reconstructions on test set
visualize_reconstructions(model, test_loader)


# Generate and visualize 20 random digits
def generate_and_visualize(model, num_images=20):
    model.eval()
    with torch.no_grad():
        # Sample random latent vectors z
        z = torch.randn(num_images, 10).to(device)
        generated_images = model.decode(z).cpu()

        # Plot the images
        n = num_images
        plt.figure(figsize=(10, 2))
        for i in range(n):
            ax = plt.subplot(2, n // 2, i + 1)
            plt.imshow(generated_images[i].squeeze(), cmap="gray")
            plt.axis("off")
        plt.suptitle("Generated Images")
        plt.tight_layout()
        plt.show()


# Call the function to generate and visualize images
generate_and_visualize(model, num_images=20)
