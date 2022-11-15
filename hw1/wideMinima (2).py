import numpy as np
import matplotlib.pyplot as plt


np.seterr(invalid='ignore', over='ignore')  # suppress warning caused by division by inf

def f(x):
    return 1/(1 + np.exp(3*(x-3))) * 10 * x**2  + 1 / (1 + np.exp(-3*(x-3))) * (0.5*(x-10)**2 + 50)

def fprime(x):
    return 1 / (1 + np.exp((-3)*(x-3))) * (x-10) + 1/(1 + np.exp(3*(x-3))) * 20 * x + (3* np.exp(9))/(np.exp(9-1.5*x) + np.exp(1.5*x))**2 * ((0.5*(x-10)**2 + 50) - 10 * x**2) 

x = np.linspace(-5,20,100)
plt.plot(x,f(x), 'k')
plt.show()



def test(lr, test_num, iter_num):
    
    for _ in range(test_num):
        # n one gradient descent
        x = np.random.rand() * 25 -5
        ylist = [x]
        xlist = np.arange(iter_num+1)
        for rep in range(iter_num):
            x = x - lr  * fprime(x)
            ylist.append(x)

        plt.plot(xlist, ylist)
    plt.show()
        
            
            
            
test(0.01, 100, 1000)
test(0.3, 100, 1000)
test(4, 100 , 1000)
        
        

    