
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
h=0
df = pd.read_csv("linear_regression_dataset.csv")
X = np.vstack((np.asarray(df.AGE.values), \
               np.asarray(df.FEMALE.values), \
               np.asarray(df.LOS.values), \
               np.asarray(df.APRDRG.values)))
y = np.asarray(df.TOTCHG.values)
def ScalingNormalization(X):
    X_norm = X
    avg = np.zeros(X.shape[1])
    max_min = np.zeros(X.shape[1])
    avg = np.vstack((X[0].mean(), X[1].mean(),X[2].mean(),X[3].mean()))
    max_min = np.vstack((X[0].std(ddof=1), X[1].std(ddof=1), X[2].std(ddof=1), X[3].std(ddof=1)))
    m = X.shape[1]
    avg_matrix = np.multiply(np.ones(m), avg).T 
    max_min_matrix = np.multiply(np.ones(m), max_min).T
    X_norm = np.subtract(X, avg).T
    X_norm = X_norm /max_min.T
    return [X_norm, avg, max_min]

Normalizeresults = ScalingNormalization(X)
X = np.asarray(Normalizeresults[0]).T
avg = Normalizeresults[1]
max_min = Normalizeresults[2]
m = len(y) 
X = np.vstack((np.ones(m), X)).T
theta = np.asarray([0,0,0,0,0]).astype(float)
iterations = 10000
alpha = 0.000001
def calcCostFunction(X, y, theta):
    m = len(y) 
    J = 0 
    h = np.sum(np.multiply(X, theta))
    SquaredError = np.power(np.subtract(h,y), 2)
    J = 1/(2*m) * np.sum(SquaredError)
    return J
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y) 
    J_history = [0,0,0,0,0]
    Theta_history = [0,0,0,0,0]
    theta_new = [0,0,0,0,0]
    for i in range(num_iters):

        for para in range(5):
            h = np.dot(X,theta.T) 
            error = np.sum(np.subtract(h, y))
            diff=np.dot(np.sum(X[para]),error)
            theta_new[para] = alpha * 1/m * diff
            theta[para] = np.subtract(theta[para], theta_new[para])  
            Theta_history.append(theta[para].tolist())
            J_history.append(calcCostFunction(X,y,theta[para]).tolist())
    return theta, Theta_history, J_history
        
results = (gradientDescent(X, y, theta, alpha, iterations))
theta = results[0]
Theta_history = results[1]
J_history = results[2]
def j_min(J_history):
    min=0
    for i in J_history:
        if i>0:
            if i<min:
                min=i;
    return i
plt.plot(J_history[0:len(J_history)], color='blue', linewidth=1)

# Put labels
plt.xlabel("Iterations")
plt.ylabel("J")