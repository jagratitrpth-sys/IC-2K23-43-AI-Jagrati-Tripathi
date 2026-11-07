import torch
import torch.nn as nn
import torch.optim as optim

def pytorch_tensors():
    print("--- 1. PyTorch Basics (Tensors) ---")
    # Tensors are the fundamental data structure in PyTorch (similar to NumPy arrays but GPU-accelerated)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.ones(2, 2)
    z = x + y
    
    print(f"Tensor X:\n{x}")
    print(f"Tensor Y (Ones):\n{y}")
    print(f"Addition (X + Y):\n{z}\n")

def linear_regression():
    print("--- 2. Linear Regression via Gradient Descent ---")
    # Toy dataset modeling the linear equation: y = 2x + 1
    X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)

    # Define a single linear layer (1 input feature, 1 output)
    model = nn.Linear(in_features=1, out_features=1)
    
    # Mean Squared Error (MSE) for loss and Stochastic Gradient Descent (SGD) for optimization
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print("Training Model for 200 epochs...")
    for epoch in range(200):
        # 1. Forward Pass: Compute predicted y
        predictions = model(X_train)
        
        # 2. Compute Loss
        loss = criterion(predictions, y_train)
        
        # 3. Backward Pass: Compute gradients
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Backpropagation
        
        # 4. Update Weights
        optimizer.step()

    print(f"Final Training Loss: {loss.item():.4f}")
    
    # Testing the trained model
    test_val = torch.tensor([[5.0]])
    prediction = model(test_val)
    print(f"Prediction for x=5 (Expected exactly 11): {prediction.item():.4f}\n")

def neural_network():
    print("--- 3. Neural Network Fundamentals ---")
    
    # Define a basic Feedforward Neural Network using PyTorch's Module class
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            # Input layer (2 features) to Hidden Layer (4 neurons)
            self.hidden = nn.Linear(2, 4) 
            # Non-linear activation function
            self.relu = nn.ReLU()
            # Hidden Layer (4 neurons) to Output Layer (1 neuron)
            self.output = nn.Linear(4, 1) 
            # Sigmoid activation for binary classification output (0 to 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.hidden(x)
            x = self.relu(x)
            x = self.output(x)
            x = self.sigmoid(x)
            return x

    # Instantiate and display the network architecture
    net = SimpleNN()
    print("Network Architecture:")
    print(net)
    
    # Pass a dummy tensor through the network
    sample_input = torch.tensor([[0.5, -0.2]])
    out = net(sample_input)
    print(f"\nForward Pass Output for input [0.5, -0.2]: {out.item():.4f}")

if __name__ == "__main__":
    pytorch_tensors()
    linear_regression()
    neural_network()