import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def compute_cost(X,y,w,b):
    m,n = X.shape
    cost = 0
    y_pred = np.zeros(m)
    for i in range(m):
        y_pred[i] = _sigmoid(np.dot(X[i],w)+b)
        
    total_cost = np.sum(- y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))
        
    total_cost = total_cost/m
    return total_cost

def compute_gradient(X, y, w, b,):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0. 
    err  = _sigmoid(np.dot(X, w) + b) - y
    for i in range(m):
        z_wb = 0
        for j in range(n): 
            z_wb += X[i,j]* w[j]   
        z_wb += b
        f_wb = _sigmoid(z_wb)
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        for j in range(n):
            dj_dw_ij = (f_wb - y[i]) * X[i][j] 
            dj_dw[j] += dj_dw_ij           
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in,alpha, num_iters):
    m = len(X)
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient(X, y, w_in, b_in)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  compute_cost(X, y, w_in, b_in)
            print(cost)
        
    return w_in, b_in

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    m,n=X.shape
    w_in = np.zeros(n)
    b_in = 0
    W,b = gradient_descent(X,y,w_in,b_in,lr,steps)
    return W,b