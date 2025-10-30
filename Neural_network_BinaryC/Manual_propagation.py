import numpy as np

#Setup
np.random.seed(42)
lr = 0.1
eps = 1e-5

# One sample
x = np.array([[0.2], [-0.4], [0.1]]) 
y = 1

# Initialize parameters
W1 = np.random.randn(4, 3) * 0.1
b1 = np.zeros((4, 1))
W2 = np.random.randn(1, 4) * 0.1
b2 = np.zeros((1, 1))


def forward(x, y, params):
    W1, b1, W2, b2 = params
    z1 = W1 @ x + b1
    a1 = np.maximum(0, z1)  
    z2 = W2 @ a1 + b2
    a2 = 1 / (1 + np.exp(-z2))  
    loss = - (y * np.log(a2) + (1 - y) * np.log(1 - a2))
    cache = (x, y, z1, a1, z2, a2)
    return loss.item(), cache

def backward(params, cache):
    W1, b1, W2, b2 = params
    x, y, z1, a1, z2, a2 = cache

    dz2 = a2 - y                          
    dW2 = dz2 @ a1.T                     
    db2 = dz2                            

    da1 = W2.T @ dz2                      
    dz1 = da1 * (z1 > 0)                  
    dW1 = dz1 @ x.T                      
    db1 = dz1                             

    grads = (dW1, db1, dW2, db2)
    return grads

def numerical_gradient(param_name, params, cache):
    W1, b1, W2, b2 = params
    num_grads = {}
    for name, W in zip(['W1', 'W2'], [W1, W2]):
        grad_approx = np.zeros_like(W)
        it = np.nditer(W, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = W[idx]
            W[idx] = old_val + eps
            L1, _ = forward(x, y, (W1, b1, W2, b2))
            W[idx] = old_val - eps
            L2, _ = forward(x, y, (W1, b1, W2, b2))
            grad_approx[idx] = (L1 - L2) / (2 * eps)
            W[idx] = old_val
            it.iternext()
        num_grads[name] = grad_approx
    return num_grads

params = (W1, b1, W2, b2)
init_loss, cache = forward(x, y, params)
print("Initial loss:", init_loss)

grads = backward(params, cache)
dW1, db1, dW2, db2 = grads
print("\nAnalytic dW1:\n", dW1)
print("\nAnalytic dW2:\n", dW2)

num_grads = numerical_gradient('W', params, cache)
max_diff_W1 = np.max(np.abs(num_grads['W1'] - dW1))
max_diff_W2 = np.max(np.abs(num_grads['W2'] - dW2))
print("\nMax abs diff (W1):", max_diff_W1)
print("Max abs diff (W2):", max_diff_W2)

W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2
params = (W1, b1, W2, b2)

final_loss, _ = forward(x, y, params)
print("\nUpdated W1:\n", W1)
print("\nUpdated W2:\n", W2)
print("\nFinal loss:", final_loss)
