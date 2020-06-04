#!/usr/bin/python3

#python lbfgsb works with numpy arrays 
import numpy as np
import python_lbfgsb as lbfgsb

#First example: min x^Tx such that x>=1. Optimal solution hits the boundary.
def quad(x):
    return x.dot(x)

def grad_quad(x):
    return x

l = np.array([1,1])
u = np.array([np.inf, np.inf])
x0 = np.array([5, 5])

print("====================== Quadratic example ======================")
opt_quad = lbfgsb.L_BFGS_B(x0, quad, grad_quad, l, u)
print("")
print("")
xOpt = np.array([1, 1])
theoOpt = {'x': xOpt, 'f': quad(xOpt), 'df': grad_quad(xOpt)}
print("Theoretical optimal value: ")
print(theoOpt)
print("Optimal value found: ")
print(opt_quad)


print("")
print("")
print("")
print("")
#Second example : min Rosenbrock function
print("====================== Rosenbrock example ======================")
def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

def grad_rosenbrock(x):
    g = np.empty(x.size)
    g[0] = 400*x[0]*(x[0]**2-x[1]) + 2*(x[0]-1)
    g[1]= 200*(-x[0]**2 + x[1])
    return g

l = np.array([-2,-2])
u = np.array([2,2])
#x0 = np.array([0.12, 0.12])
x0 = np.array([-1, -1])

opt_rosenbrock = lbfgsb.L_BFGS_B(x0, rosenbrock, grad_rosenbrock, l, u)
print("")
print("")
theoOpt = {'x': np.array([1, 1]), 'f': 0, 'df': grad_rosenbrock(np.array([1,1]))}
print("Theoretical optimal value: ")
print(theoOpt)
print("Optimal value found: ")
print(opt_rosenbrock)


print("")
print("")
print("")
print("")
#Third example : min Beale function
print("====================== Beale example ======================")
def beale(x):
    return (1.5-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]*x[1]**3)**2

def grad_beale(x):
    y1 = x[1]
    y2 = y1*y1
    y3 = y2*y1
    f1 = 1.5 - x[0] + x[0] * y1
    f2 = 2.25 - x[0] + x[0] * y2
    f3 = 2.625 - x[0] + x[0] * y3
    
    return np.array([2*(y1 - 1)*f1 + 2*(y2 - 1)*f2 + 2*(y3 - 1)*f3,\
                     2*x[0]*f1+ 4*x[0]*y1*f2 + 6*x[0]*y2*f3])

l = -4.5*np.ones(2)
u = -l
x0 = np.array([2.5,-1.3])

opt_beale = lbfgsb.L_BFGS_B(x0, beale, grad_beale, l, u, epsf = 1e1, epsg = 1e-10)
print("")
print("")
theoOpt = {'x': np.array([3, 0.5]), 'f': beale(np.array([3, 0.5])), 'df': grad_beale(np.array([3,0.5]))}
print("Theoretical optimal value: ")
print(theoOpt)
print("Optimal value found: ")
print(opt_beale)
