
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y):
        loss, portfolio_value = self.criterion(x, y)
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss, portfolio_value


class SimpleLossCompute_tst:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y):
        if self.opt is not None:
            loss, portfolio_value = self.criterion(x, y)
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
            return loss, portfolio_value
        else:
            loss, portfolio_value, SR, CR, St_v, tst_pc_array, TO = self.criterion(x, y)
            return loss, portfolio_value, SR, CR, St_v, tst_pc_array, TO
