import torch
from torch import nn


class Batch_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1, beta=0.1, size_average=True, device="cpu"):
        super(Batch_Loss, self).__init__()
        self.gamma = gamma  # variance penalty
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio = commission_ratio
        self.interest_rate = interest_rate
        self.device = device

    def forward(self, w, y):  # w:[128,1,12]   y:[128,11,4]
        close_price = y[:, :, 0:1].to(self.device)  # [128,11,1]
        # future close prise (including cash)
        close_price = torch.cat([torch.ones(close_price.size()[0], 1, 1).to(self.device), close_price], 1).to(self.device)  # [128,11,1]cat[128,1,1]->[128,12,1]
        reward = torch.matmul(w, close_price)  # [128,1,1]
        close_price = close_price.view(close_price.size()[0], close_price.size()[2], close_price.size()[1])  # [128,1,12]
        ###############################################################################################################
        element_reward = w * close_price
        interest = torch.zeros(element_reward.size(), dtype=torch.float).to(self.device)
        interest[element_reward < 0] = element_reward[element_reward < 0]
        interest = torch.sum(interest, 2).unsqueeze(2) * self.interest_rate  # [128,1,1]
        ###############################################################################################################
        future_omega = w * close_price / reward  # [128,1,12]
        wt = future_omega[:-1]  # [128,1,12]
        wt1 = w[1:]  # [128,1,12]
        pure_pc = 1 - torch.sum(torch.abs(wt - wt1), -1) * self.commission_ratio  # [128,1]
        pure_pc = pure_pc.to(self.device)
        pure_pc = torch.cat([torch.ones([1, 1]).to(self.device), pure_pc], 0)
        pure_pc = pure_pc.view(pure_pc.size()[0], 1, pure_pc.size()[1])  # [128,1,1]

        cost_penalty = torch.sum(torch.abs(wt - wt1), -1)
        ################## Deduct transaction fee ##################
        reward = reward * pure_pc  # reward=pv_vector
        ################## Deduct loan interest ####################
        reward = reward + interest
        portfolio_value = torch.prod(reward, 0)
        batch_loss = -torch.log(reward)
        #####################variance_penalty##############################
        #        variance_penalty = ((torch.log(reward)-torch.log(reward).mean())**2).mean()
        if self.size_average:
            loss = batch_loss.mean()  # + self.gamma*variance_penalty + self.beta*cost_penalty.mean()
            return loss, portfolio_value[0][0]
        else:
            loss = batch_loss.mean()  # +self.gamma*variance_penalty + self.beta*cost_penalty.mean() #(dim=0)
            return loss, portfolio_value[0][0]
