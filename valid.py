import numpy as np

def cross_validation(x, y, k, model):
    n = len(x)
    if n % k != 0:
        print("haven't impl yet.")
        return
    
    interval = int(n / k)
    loss = 0
    for i in range(0,n,interval):
        train_x, train_y = np.concatenate([x[:i], x[i+interval:]], axis=0), np.concatenate([y[:i], y[i+interval:]], axis=0)
        valid_x, valid_y = x[i:i+interval], y[i:i+interval]
        model.train(train_x, train_y)
        loss += model.test(valid_x, valid_y)
        
    return loss / n