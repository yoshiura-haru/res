# coding: utf-8
import numpy as np

class SGD:

    """確率的勾配降下法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 

class CG:
    def __init__(self):
        self.A = None
        self.b = None
        self.x = None
        self.r = None
        self.p = None

    def init(self, A, b, x0):
        # 次元チェック
        assert A.ndim == 2
        assert A.shape[0] == A.shape[1]
        assert b.shape == x0.shape
        assert A.shape[0] == b.shape[0]

        self.A = A
        self.b = b
        self.x = x0.copy()

        self.r = b - A @ self.x
        self.p = self.r.copy()

    def update(self):
        Ap = self.A @ self.p

        r_dot_r = self.r @ self.r
        denom = self.p @ Ap

        # 数値安定性（SPD が壊れた場合の保険）
        if abs(denom) < 1e-12:
            return self.x

        alpha = r_dot_r / denom
        self.x += alpha * self.p

        r_new = self.r - alpha * Ap
        beta = (r_new @ r_new) / (r_dot_r + 1e-12)

        self.p = r_new + beta * self.p
        self.r = r_new

        return self.x


class CGOptimizer:
    def __init__(self, spd_data, cg_iters=5, lr=0.01):
        self.spd_data = spd_data      # (N, d, d)
        self.cg_iters = cg_iters
        self.lr = lr
        self.ptr = 0                  # SPD 行列の参照位置

    def update(self, params, grads):

        # ===== 第1層 : CG =====
        W1 = params["W1"]             # (784, 50)
        gW1 = grads["W1"]             # (784, 50)

        assert W1.ndim == 2
        assert W1.shape == gW1.shape

        A = self.spd_data[self.ptr]   # (50, 50)
        self.ptr = (self.ptr + 1) % len(self.spd_data)

        d = W1.shape[1]
        assert A.shape == (d, d)

        for i in range(W1.shape[0]):  # 784 行
            w = W1[i]                 # (50,)
            g = gW1[i]                # (50,)

            b = -g
            x0 = np.zeros_like(w)

            cg = CG()
            cg.init(A, b, x0)

            for _ in range(self.cg_iters):
                delta = cg.update()

            W1[i] = w + delta

        params["W1"] = W1

        # ===== その他の層 : SGD =====
        for key in ("b1", "W2", "b2"):
            params[key] -= self.lr * grads[key]
