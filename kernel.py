import numpy as np

# Kernel abst
class Kernel:
    def __init__(self):
        return
    
    def computeKernelVec(self, x, x_all):
        return np.array([self.__call__(x, xi) for xi in x_all])
    
    def computeKernelMat(self, x_all):
        n = len(x_all)
        K = np.empty((n, n))
        for i, xi in enumerate(x_all):
            K[i,:] = self.computeKernelVec(xi, x_all)
        return K

# Kernel impl example
class IdentityKernel:
    def __init__(self):
        super().__init__()
        return
    
    def __call__(self, x, xi):
        return x
    
# Gauss Kernel
class GaussKernel(Kernel):
    def __init__(self, h=1.0):
        super().__init__()
        self.h = h
        return
    
    def __call__(self, x, xi):
        return np.exp(-np.linalg.norm(x-xi)**2/(2*self.h**2))
    
    def setParam(self, h):
        self.h = h
        return