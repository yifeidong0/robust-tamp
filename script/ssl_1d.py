"""PaperBot implementation."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Hyperparameters
epsilon = 0.2  # Exploration probability
T = 200  # Number of total trials
N = 50   # Number of iterations for fitting surrogate model
M = 50   # Number of iterations for inverse design
lambda_z = 0.1  # Step size for design optimization
lambda_theta = 0.1  # Learning rate for surrogate model
learning_rate = 0.01

# Design space
design_space = np.linspace(0, 1, 100)

# Actual reward function (sine curve with noise)
def reward_function(z):
    return np.sin(2 * np.pi * z) + 0.1 * np.random.randn()

# Neural network surrogate model
class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.fc1 = nn.Linear(1, 32)  # First fully connected layer with 32 neurons
        self.fc2 = nn.Linear(32, 16)  # Second fully connected layer with 16 neurons
        self.fc3 = nn.Linear(16, 8)   # Third fully connected layer with 8 neurons
        self.fc4 = nn.Linear(8, 1)    # Fourth fully connected layer, final output layer
    
    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        return self.fc4(z)

# Initialize surrogate model and optimizer
surrogate_model = SurrogateModel()
optimizer = optim.Adam(surrogate_model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Helper function to train the surrogate model
def train_surrogate_model(model, optimizer, Z, R, n_iter=N):
    model.train()
    for _ in range(n_iter):
        optimizer.zero_grad()
        Z_tensor = torch.tensor(Z[:, None], dtype=torch.float32)
        R_tensor = torch.tensor(R[:, None], dtype=torch.float32)
        predictions = model(Z_tensor)
        loss = loss_fn(predictions, R_tensor)
        loss.backward()
        optimizer.step()

# Initialize variables
designs = []
rewards = []

# Main loop (SSL algorithm with epsilon-greedy exploration and inverse design)
for t in range(T):
    if np.random.rand() < epsilon:
        # Exploration step: random design
        z = np.random.choice(design_space)
    else:
        # Exploitation step: use the surrogate model to find the best design
        z_candidates = np.linspace(0, 1, M)
        z_tensor = torch.tensor(z_candidates[:, None], dtype=torch.float32)
        surrogate_predictions = surrogate_model(z_tensor).detach().numpy()
        z = z_candidates[np.argmax(surrogate_predictions)]
    
    # Evaluate reward for the chosen design
    r = reward_function(z)
    
    # Store design and reward
    designs.append(z)
    rewards.append(r)
    
    # Fit surrogate model with the updated data
    Z = np.array(designs)
    R = np.array(rewards)
    train_surrogate_model(surrogate_model, optimizer, Z, R)
    
    # Plot intermediate results
    if t % 10 == 0:
        plt.figure(figsize=(10, 6))
        
        # Plot true reward function
        true_rewards = [reward_function(z) for z in design_space]
        plt.plot(design_space, true_rewards, label="True reward function (sine with noise)", color="blue")
        
        # Plot surrogate model's predictions
        z_tensor = torch.tensor(design_space[:, None], dtype=torch.float32)
        surrogate_rewards = surrogate_model(z_tensor).detach().numpy().flatten()
        plt.plot(design_space, surrogate_rewards, label="Surrogate model predictions", color="red")
        
        # Plot sampled designs and their rewards
        plt.scatter(Z, R, color="green", label="Sampled designs")
        
        plt.xlabel("Design")
        plt.ylabel("Reward")
        plt.title(f"Iteration {t} - Surrogate Model vs True Reward")
        plt.legend()
        plt.savefig(f"ssl_training_iteration_{t}.png")
        plt.show()

# Final plot after all iterations
plt.figure(figsize=(10, 6))

# Plot true reward function
true_rewards = [reward_function(z) for z in design_space]
plt.plot(design_space, true_rewards, label="True reward function (sine with noise)", color="blue")

# Plot surrogate model's predictions
z_tensor = torch.tensor(design_space[:, None], dtype=torch.float32)
surrogate_rewards = surrogate_model(z_tensor).detach().numpy().flatten()
plt.plot(design_space, surrogate_rewards, label="Surrogate model predictions", color="red")

# Plot sampled designs and their rewards
plt.scatter(Z, R, color="green", label="Sampled designs")

# Labels and title
plt.xlabel("Design")
plt.ylabel("Reward")
plt.title(f"Final Surrogate Model vs True Reward")
plt.legend()

# Save the final plot
plt.savefig("final_ssl_training_result.png")
plt.show()