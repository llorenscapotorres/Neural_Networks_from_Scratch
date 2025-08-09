import numpy as np

# --- Activation Functions ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis = 1, keepdims=True)) # Numeric stability
    return exp_x / np.sum(exp_x, axis = 1, keepdims=True)

# --- Loss MSE ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return (2 * (y_pred - y_true)) / y_true.shape[0]

# --- Softmax's Jacobian for a vector ---
def softmax_jacobian(s):
    s = s.reshape(-1, 1) # column
    return np.diagflat(s) - np.dot(s, s.T)

# --- Neural Network ---
class NeuralNetwork:
    """
    At the moment, only one hidden layer. Soon it will change.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01 # 784 x 10
        self.b1 = np.zeros((1, hidden_size)) # 1 x 10
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01 # 10 x 10
        self.b2 = np.zeros((1, output_size)) # 1 x 10

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1 # m x 10
        self.a1 = relu(self.z1) # m x 10
        self.z2 = self.a1 @ self.W2 + self.b2 # m x 10
        self.a2 = softmax(self.z2) # m x 10
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        grad_z2 = np.zeros_like(self.a2) # m x 10

        # Compute the jacobian for each sample
        for i in range(m):
            da2 = mse_derivative(y[i], self.a2[i]) # dLoss / da2
            J = softmax_jacobian(self.a2[i]) # Jacobian --> 10 x 10
            grad_z2[i] = J @ da2 # dLoss / dz2

        dW2 = (self.a1.T @ grad_z2) / m # 10 x 10
        db2 = np.sum(grad_z2, axis = 0, keepdims = True) / m

        dz1 = (grad_z2 @ self.W2.T) * relu_derivative(self.z1) # m x 10
        dW1 = (X.T @ dz1) / m # 784 x 10
        db1 = np.sum(dz1, axis = 0, keepdims = True) / m

        # --- Update ---
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=1000, batch_size=32):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            # Compute loss in all samples
            loss = mse(y, self.forward(X))
            if epoch % 1 == 0:
                print(f"Epoch {epoch} - Loss: {loss:.6f}")