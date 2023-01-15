class Dataset:
    def __init__(self, X, w, Y_0, Y_1, Y_obs):
        self.X = X
        self.w = w
        self.Y_0 = Y_0
        self.Y_1 = Y_1
        self.Y_obs = Y_obs