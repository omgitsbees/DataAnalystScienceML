import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the MAML algorithm
class MAML:
    def __init__(self, model, inner_lr, outer_lr, num_inner_steps):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)
    
    def inner_loop(self, x_train, y_train):
        # Clone the model parameters for inner loop optimization
        temp_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for _ in range(self.num_inner_steps):
            y_pred = self.model(x_train)
            loss = nn.CrossEntropyLoss()(y_pred, y_train)
            grads = grad(loss, temp_weights.values(), create_graph=True)
            temp_weights = {name: temp_weights[name] - self.inner_lr * grad for (name, grad) in zip(temp_weights.keys(), grads)}
        
        return temp_weights
    
    def outer_loop(self, tasks):
        meta_loss = 0.0
        
        for x_train, y_train, x_val, y_val in tasks:
            # Inner loop
            temp_weights = self.inner_loop(x_train, y_train)
            
            # Update the model with the temporary weights
            y_pred = self.model.forward(x_val)
            loss = nn.CrossEntropyLoss()(y_pred, y_val)
            meta_loss += loss
        
        # Outer loop optimization
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
    
    def adapt(self, x_train, y_train):
        # Perform a few steps of gradient descent to adapt to a new task
        for _ in range(self.num_inner_steps):
            y_pred = self.model(x_train)
            loss = nn.CrossEntropyLoss()(y_pred, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Function to generate synthetic tasks
def generate_synthetic_task():
    # Generates a synthetic task: (x_train, y_train, x_val, y_val)
    x_train = torch.randn(5, 1).to(device)
    y_train = torch.randint(0, 2, (5,)).to(device)
    x_val = torch.randn(5, 1).to(device)
    y_val = torch.randint(0, 2, (5,)).to(device)
    return x_train, y_train, x_val, y_val

# Main script to run the MAML process
if __name__ == "__main__":
    # Initialize model and MAML
    input_size = 1
    hidden_size = 10
    output_size = 2
    model = SimpleNN(input_size, hidden_size, output_size)
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=5)

    # Train the meta-learner
    num_meta_iterations = 1000
    for iteration in range(num_meta_iterations):
        tasks = [generate_synthetic_task() for _ in range(4)]  # 4 tasks per meta-update
        maml.outer_loop(tasks)
        if iteration % 100 == 0:
            print(f'Iteration {iteration}: Meta-learning step completed.')

    # Example of adapting to a new task
    x_train, y_train, _, _ = generate_synthetic_task()
    maml.adapt(x_train, y_train)

    # Test on validation data (assuming it's available)
    x_test = torch.randn(5, 1).to(device)
    y_test = torch.randint(0, 2, (5,)).to(device)
    y_pred = model(x_test)
    test_loss = nn.CrossEntropyLoss()(y_pred, y_test)
    print(f'Test Loss: {test_loss.item()}')
