class Trigger:
    def __init__(self, minimize=True):
        self.minimize = minimize
        if self.minimize:
            self.min = None
        else:
            self.max = None
        
    def is_best(self, val):
        if self.minimize:
            if self.min is None or val <= self.min:
                self.min = val
                return True
            return False
        else:
            if self.max is None or val >= self.max:
                self.max = val
                return True
            return False