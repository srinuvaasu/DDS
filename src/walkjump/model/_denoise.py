import torch
from torch import nn

from walkjump.data import AbBatch
from walkjump.utils import isotropic_gaussian_noise_like

from ._base import TrainableScoreModel


class DenoiseModel(TrainableScoreModel):
    needs_gradients: bool = False

    def score(self, y: torch.Tensor) -> torch.Tensor:
        
        #self.sigma = self.sigma.view(-1, 1, 1)
        
        #return (self.model(y,self.sigma) - y) / pow(self.sigma, 2)
        #print(self.sigma)
        return (self.model(y) - y) / pow(self.sigma, 2)

    def compute_loss(self, batch: AbBatch,Sigma=None) -> torch.Tensor:
        if Sigma==None:
            y = batch.x + isotropic_gaussian_noise_like(batch.x, self.sigma)    
            #print("none")
        else:
            while Sigma.dim() < batch.x.dim():
                Sigma = Sigma.unsqueeze(-1)
            y = batch.x +torch.randn_like(batch.x)*Sigma
        #y = batch.x + isotropic_gaussian_noise_like(batch.x, self.sigma)
        xhat = self.xhat(y)
        return nn.MSELoss()(xhat, batch.x)
