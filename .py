import numpy as np

data = np.loadtxt('data_HL_simple.txt', usecols=range(5))
print("Original data")
print(data)

print("\nCorrelation matrix computed using NumPy")
print(np.corrcoef(data.transpose()))

def mycorr(X, i, k):
    xi = X[:, i]
    xk = X[:, k]
    xi_mean = np.mean(xi)
    xk_mean = np.mean(xk)
    numerator = np.sum((xi - xi_mean) * (xk - xk_mean))
    denominator = np.sqrt(np.sum((xi - xi_mean) ** 2) * np.sum((xk - xk_mean) ** 2))
    return numerator / denominator

n = data.shape[1]
corr_matrix_manual = np.zeros((n, n))
for i in range(n):
    for k in range(n):
        corr_matrix_manual[i, k] = mycorr(data, i, k)

print("\nCorrelation matrix computed manually")
print(corr_matrix_manual)

data_centered = data - np.mean(data, axis=0)
scatter_matrix = np.dot(data_centered.T, data_centered)
D = np.sqrt(np.diag(scatter_matrix))
pearson_r = scatter_matrix / (D[:, None] * D[None, :])

print("\nCorrelation matrix computed using matrix algebra")
print(pearson_r)
