import numpy as np
import kernel as knl

# 
class LinearKernelModel:
    def __init__(self):
        return
    
    def __call__(self, kernel, x):
        return np.array([np.dot(self.param, kernel.computeKernelVec(xi, self.x_all)) for xi in x])
    
    def setParam(self, param, x_all):
        self.param = param
        self.x_all = x_all
        return

class L2SquareLoss:
    def __init__(self, l=1.0):
        self.l = l
        return
    
    def __call__(self, model, kernel, x, y):
        return (np.linalg.norm(model(kernel, x) - y)**2 + self.l * np.linalg.norm(model.param)**2)/2.0
    
    def setParam(self, l):
        self.l = l

# a whole model which has linear model, kernel, and criteria
class L2LinearGaussKernelModel:
    def __init__(self):
        self.criteria = L2SquareLoss()
        self.model = LinearKernelModel()
        self.kernel = knl.GaussKernel()
        return
    
    def __call__(self, x):
        return self.model(self.kernel, x)
        
    def train(self, x, y):
        K = self.kernel.computeKernelMat(x)
        U_inv = np.linalg.inv(np.dot(K.T, K) + self.criteria.l * np.eye(len(x)))
        self.model.setParam(np.dot(np.dot(U_inv, K.T), y), x)
        return
    
    def test(self, x, y):
        return self.criteria(self.model, self.kernel, x, y)
    
    def setParam(self, kernel_h, l2_lambda):
        self.kernel.setParam(kernel_h)
        self.criteria.setParam(l2_lambda)