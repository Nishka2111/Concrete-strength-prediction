import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn


# Loading the Concrete_Data.csv dataset
data = pd.read_csv("Concrete_Data.csv")

# Show shape of the dataset
print("Shape of the dataset:", data.shape)

# Display first few rows
data.head()
dataset_shape = data.shape
column_names = data.columns.tolist()

# Display results
print("Dataset shape:", dataset_shape)
print("Column names:", column_names)

# Inputs: all columns except the last one
inputs = data.iloc[:, :-1]

# Targets: only the last column
targets = data.iloc[:, -1]

print("Input shape:", inputs.shape)
print("Target shape:", targets.shape)

import torch

# Convert inputs and targets to PyTorch tensors
inputs_tensor = torch.tensor(inputs.values, dtype=torch.float32)
targets_tensor = torch.tensor(targets.values, dtype=torch.float32)

# Check their shapes
print("Inputs Tensor Shape:", inputs_tensor.shape)
print("Targets Tensor Shape:", targets_tensor.shape)

# Slice the first 100 rows
inputs_100 = inputs_tensor[:100]
targets_100 = targets_tensor[:100]

# Print shapes
print("Shape of inputs_100:", inputs_100.shape)
print("Shape of targets_100:", targets_100.shape)

import matplotlib.pyplot as plt

#histogram (Concrete Strength)
plt.hist(targets, bins=30, edgecolor='k')
plt.title('Distribution of Concrete Strength')
plt.xlabel('Strength (MPa)')
plt.ylabel('Count')
plt.show()

water_feature = data['Water  (component 4)(kg in a m^3 mixture)']

#histogram
plt.figure()
plt.hist(water_feature , bins=10)
plt.xlabel("Water (kg in a m^3 mixture)")
plt.title("Histogram of Water (Component 4)")
plt.show()

# Scatterplot: Cement vs Concrete Strength
plt.scatter(inputs['Cement (component 1)(kg in a m^3 mixture)'], targets, alpha=0.6)
plt.title('Cement vs Concrete Strength')
plt.xlabel('Cement (kg/m^3)')
plt.ylabel('Strength (MPa)')
plt.show()

# 1. Load dataset
data = pd.read_csv("Concrete_Data.csv")  # Adjust path as needed

# 2. Separate features and target
inputs = data.iloc[:, :-1].values       # First 8 columns = features
targets = data.iloc[:, -1].values.reshape(-1, 1)  # Last column = target

# 3. Normalize input features
scaler = StandardScaler()
inputs_scaled = scaler.fit_transform(inputs)

# 4. Convert to PyTorch tensors
inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)

# 5. Shuffle and split into training (80%) and test (20%) sets
torch.manual_seed(42)
n_samples = inputs_tensor.shape[0]
indices = torch.randperm(n_samples)
split_idx = int(n_samples * 0.8)

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = inputs_tensor[train_indices]
y_train = targets_tensor[train_indices]
X_test = inputs_tensor[test_indices]
y_test = targets_tensor[test_indices]

# 6. Initialize model parameters
num_features = X_train.shape[1]  # Should be 8
W = torch.randn((num_features, 1), requires_grad=True)
W.data *= 0.01  # Scale down initial values
b = torch.zeros((1,), requires_grad=True)

# 7. Confirm everything is set up
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])
print("Weight shape:", W.shape)
print("Bias shape:", b.shape)


import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

class CustomMLPModel(nn.Module):
    def __init__(self):
        super(CustomMLPModel, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(8, 24)   
        self.relu1 = nn.ReLU()     
        self.fc2 = nn.Linear(24, 12) 
        self.relu2 = nn.ReLU()      
        self.fc3 = nn.Linear(12, 1)   
    # Step 2: Define the forward pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Step 3: Instantiate and print the model
custom_model = CustomMLPModel()
print(custom_model)

# Load the Concrete dataset
data = pd.read_csv("Concrete_Data.csv")

# Separate features and target
X = data.iloc[:, :-1].values  # First 8 columns
y = data.iloc[:, -1].values.reshape(-1, 1)  # Last column (strength)

# Standardize inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Set seed and shuffle
torch.manual_seed(42)
num_samples = X_tensor.shape[0]
indices = torch.randperm(num_samples)

# 80/20 split
split_idx = int(0.8 * num_samples)
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

# Create splits
X_train = X_tensor[train_idx]
y_train = y_tensor[train_idx]
X_test = X_tensor[test_idx]
y_test = y_tensor[test_idx]

# Print shapes
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

import torch.optim as optim

# Instantiate the model
model = CustomMLPModel()

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(model)
print(optimizer)

# Number of training epochs
num_epochs = 300

# List to store loss values per epoch
train_losses = []

# Training loop
for epoch in range(num_epochs):
    # ===== Forward Pass =====
    y_pred = model(X_train)

    # ===== Compute Loss =====
    loss = loss_fn(y_pred, y_train)

    # ===== Backward Pass =====
    loss.backward()

    # ===== Optimizer Step =====
    optimizer.step()

    # ===== Zero Gradients =====
    optimizer.zero_grad()

    # ===== Record Loss =====
    train_losses.append(loss.item())

    # ===== Print Occasionally =====
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Plot the training loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(num_epochs), train_losses, linestyle='-', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Training Loss (MSE)')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.show()


model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = loss_fn(test_pred, y_test)

print("Test MSE:", test_loss.item())
