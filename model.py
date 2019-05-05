import numpy as np
import kernel as knl

# 
class LinearKernelModel:
    def __init__(self, param):
        self.param = param
        return
    
    def __call__(self, kernel, x, x_all):
        return np.dot(self.param, kernel.computeKernelVec(x, x_all))
    
    def setParam(self, param):
        self.param = param
        return

class L2SquareLoss:
    def __init__(self, l=1.0):
        self.l = l
        return
    
    def __call__(self, param, t, y):
        return np.sum((np.linalg.norm(t - y)**2 + self.l * np.linalg.norm(param)**2))/2.0
    
    def setParam(self, l):
        self.l = l
        
class L2KernelLearner:
    def __init__(self):
        return
    
    def __call__(self, x, y, kernel, l):
        K = kernel.computeKernelMat(x)
        U_inv = np.linalg.inv(np.dot(K.T, K) + l * np.eye(len(x)))
        return np.dot(np.dot(U_inv, K.T), y)

# a whole model which has linear model, kernel, and criteria
class L2LinearKernelModel:
    def __init__(self, kernel, param):
        self.model = LinearKernelModel(param)
        self.learner = L2KernelLearner()
        self.criteria = L2SquareLoss()        
        self.kernel = kernel #knl.GaussKernel()
        return
    
    def __call__(self, x):
        return np.array([self.model(self.kernel, xi, self.x_all) for xi in x])
        
    def train(self, x, y):
        self.x_all = x
        self.model.setParam(self.learner(x, y, self.kernel, self.criteria.l))
    
    def test(self, x, y):
        return self.criteria(self.model.param, self.__call__(x), y)
    
    def setKernelParams(self, params):
        self.kernel.setParams(params)
        return
    
    def setCriteriaParams(self, params):
        self.criteria.setParam(params)
        return
            
####################################
## kadai2 2019/05/07 L1 sparse reg
####################################
class L1Loss:
    def __init__(self, l = 0.1):
        self.l = l
        
    def __call__(self, param, t, y):
        return np.sum((t-y)**2)/2.0 + self.l * np.linalg.norm(param, ord=1)
        
    def setParam(self, l):
        self.l = l
        
class L1GaussKernelADMMModel:
    def __init__(self, theta, z, u, param, l):
        self.theta = theta
        self.z = z
        self.u = u
        self.kernel = knl.GaussKernel()
        self.model = LinearKernelModel(param)
        self.criteria = L1Loss(l)
        
    def __call__(self, x):
        return np.array([self.model(self.kernel, xi, self.x_all) for xi in x])
        
    def updateOnce(self, x, y):
        K = self.kernel.computeKernelMat(x)
        KKI_inv = np.linalg.inv(np.dot(K.T, K) + np.eye(x.shape[0]))
        B = np.dot(K.T, y) - self.u + self.z
        self.theta = np.dot(KKI_inv, B)
        print(KKI_inv.shape, B.shape, self.theta.shape)
        self.z = np.array([L1GaussKernelADMMModel.softThresholdProcess(theta_i, u_i, self.criteria.l)
            for theta_i, u_i in zip(self.theta, self.u)])
        self.u = self.u + self.theta - self.z
        
    def train(self, x, y):
        self.x_all = x
        self.updateOnce(x, y)
        self.model.setParam(self.theta)
            
    def test(self, x, y):
        return self.criteria(self.model.param, self.__call__(x), y)
            
    @staticmethod
    def softThresholdProcess(theta, u, l):
        return max(0, theta+u-l) + max(0, -theta-u-l)