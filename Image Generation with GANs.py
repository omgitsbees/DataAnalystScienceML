import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
image_size = 784
hidden_size = 256
latent_size = 100
batch_size = 100
num_epochs = 100

# Data augmentation and normalization for training
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)])

# Load Fashion MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, image_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(image_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Loss function and optimizer
criterion = nn.BCELoss()
generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizer_g = torch.optim.AdamW(generator.parameters(), lr=0.001)
optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=0.001)

# Training
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        # Adversarial ground truths
        valid = torch.ones((images.size(0), 1), dtype=torch.float32, device=device)
        fake = torch.zeros((images.size(0), 1), dtype=torch.float32, device=device)

        # Configure input
        images = images.view(-1, image_size).to(device)

        # Sample noise as generator input
        noise = torch.randn((images.size(0), latent_size), device=device)

        # Generate a batch of images
        gen_images = generator(noise)

        # Train discriminator
        optimizer_d.zero_grad()
        d_loss = (criterion(discriminator(images), valid) + criterion(discriminator(gen_images.detach()), fake)) / 2
        d_loss.backward()
        optimizer_d.step()

        # Train generator
        optimizer_g.zero_grad()
        g_loss = criterion(discriminator(gen_images), valid)
        g_loss.backward()
        optimizer_g.step()

    # Print losses
    print(f'Epoch [{epoch+1}/{num_epochs}], D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}')

# Generate some samples
sample_noise = torch.randn((25, latent_size), device=device)
samples = generator(sample_noise)
samples = samples.view(-1, 1, 28, 28).cpu().detach().numpy()

# Plot the samples
plt.figure(figsize=(5, 5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(samples[i], cmap='gray')
    plt.axis('off')
plt.show()