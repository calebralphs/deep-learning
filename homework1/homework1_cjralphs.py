import numpy as np

def problem_a (A, B):
    return A + B

def problem_b (A, B, C):
    return np.dot(A,B) - C

def problem_c (A, B, C):
    return A*B + C.T

def problem_d (x, y):
    return np.dot(x.T,y)

def problem_e (A):
    return np.zeros(A.shape)

def problem_f (A, x):
    return np.linalg.solve(A,x)

def problem_g (A, x):
    return np.linalg.solve(A.T,x.T).T

def problem_h (A, alpha):
    return A + alpha*np.eye(len(A))

def problem_i (A, i, j):
    return A[i, j]

def problem_j (A, i):
    return np.sum(A[i])

def problem_k (A, c, d):
    return np.mean(A[(A >= c) & (A <= d)])

def problem_l (A, k):
    eig_vals, eig_vecs = np.linalg.eig(A)
    eig_vals_idx = np.argsort(eig_vals)[::-1][:k]
    return eig_vecs[eig_vals_idx]

def problem_m (x, k, m, s):
    return np.random.multivariate_normal(
        x+m*np.ones(x.shape[0]),
        s*np.identity(x.shape[0]),
        (x.shape[0],k))

def problem_n (A):
    return np.random.shuffle(A)


def linear_regression (X_tr, y_tr):
    w = np.dot(np.linalg.solve(np.dot(X_tr.T,X_tr), X_tr.T),y_tr)
    return w

def fMSE (w, X, y):
    yhat = np.dot(X,w)
    MSE = np.sum(np.square(yhat-y)) / (2*y.shape[0])
    return MSE

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, y_tr)

    # Report fMSE cost on the training and testing data (separately)
    MSE_tr = fMSE(w, X_tr, y_tr)
    MSE_te = fMSE(w, X_te, y_te)
    print(MSE_tr)
    print(MSE_te)

if __name__ == "__main__":
    train_age_regressor()
