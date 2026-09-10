class LionOptimizer:
    """
    Lion (EvoLved Sign Momentum) Optimizer.
    Chen et al. (Google Brain, 2023).
    Only tracks momentum, computes update as sign(beta1 * m + (1 - beta1) * g).
    """
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0.01):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.wd = weight_decay
        self.exp_avg = {}

    def _sign(self, val):
        return 1.0 if val > 0 else (-1.0 if val < 0 else 0.0)

    def step(self, param, grad, param_id=0):
        if param_id not in self.exp_avg:
            self.exp_avg[param_id] = 0.0

        m = self.exp_avg[param_id]
        update_dir = self._sign(self.b1 * m + (1.0 - self.b1) * grad)
        
        # Apply decoupled weight decay
        param = param - self.lr * self.wd * param
        # Apply sign update
        param = param - self.lr * update_dir
        # Update EMA momentum
        self.exp_avg[param_id] = self.b2 * m + (1.0 - self.b2) * grad
        return param
