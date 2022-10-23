import matplotlib.pyplot as plt 
import numpy as np 


"""
Step 1 : Generate Toy data
"""

d = 35
n_train, n_val, n_test = 300, 60, 30
np.random.seed(0)
beta = np.random.randn(d)
beta_true = beta / np.linalg.norm(beta)
# Generate and fix training data
X_train = np.array([np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(n_train)])
Y_train = X_train @ beta_true + np.random.normal(loc = 0.0, scale = 0.5, size = n_train)
# Generate and fix validation data (for tuning lambda). 
X_val = np.array([np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(n_val)])
Y_val = X_val @ beta_true 
# Generate and fix test data
X_test = np.array([np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(n_test)])
Y_test = X_test @ beta_true 
 

"""
Step 2 : Solve the problem
"""
def f(X,W):
    L=[]
    for i in range(len(X)):
        L.append((W @ X[i]) * (W @ X[i]>0))
    return np.array(L)

fixed_lambda = 0.01
lambda_list = [2 ** i for i in range(-6, 6)]
num_params = np.arange(1,1501,10)

errors_opt_lambda = []
errors_fixed_lambda = []
for p in num_params : 
    # fix W, calculate Xtilda based on fixed W
    W=np.random.normal(0, 1/np.sqrt(p), size=(p,d))
    X_tilda = f(X_train, W)
    X_val_tilda = f(X_val, W)
    X_test_tilda = f(X_test, W)
    
    # theta based on fixed lambda
    theta_fixed = np.linalg.inv(X_tilda.T @ X_tilda + fixed_lambda * np.identity(p) )@ X_tilda.T @ Y_train
    
    # find optimal lambda using validation data
    minloss = 10000000000000
    theta_optimal = np.zeros(p)
    for l in lambda_list:
        theta =  np.linalg.inv(X_tilda.T @ X_tilda + l * np.identity(p) )@ X_tilda.T @ Y_train
        loss = np.linalg.norm(X_val_tilda@theta - Y_val)
        if(loss < minloss):
            minloss = loss
            theta_optimal = theta
    
    
    #use test set to calculate accuracy
    fixed_loss = np.linalg.norm(X_test_tilda@theta_fixed - Y_test)
    errors_fixed_lambda.append(fixed_loss)
        
    optimal_loss = np.linalg.norm(X_test_tilda@theta_optimal - Y_test)
    errors_opt_lambda.append(optimal_loss)
    
    #just for debugging
    print(p, " done")

"""
Step 3 : Plot the results
"""    

plt.figure(figsize = (24, 8))
plt.rc('text', usetex = False)
plt.rc('font', family = 'serif')
plt.rc('font', size = 24)


plt.scatter(num_params, errors_fixed_lambda, color = 'black',
            label = r"Test error with fixed $\lambda = 0.01$",
            ) 
plt.legend()

plt.plot(num_params, errors_opt_lambda, 'k', label = r"Test error with tuned $\lambda$")
plt.legend()
plt.xlabel(r'$\#$ parameters')
plt.ylabel('Test error')
plt.title(r'Test error vs. $\#$ params : SeoyongLee')

plt.savefig('double_descent.png')
plt.show()