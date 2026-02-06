import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.revin import RevIN
import math



class FreNetwork(nn.Module):
    def __init__(self, configs):
        super(FreNetwork, self).__init__()
        self.embed_size = configs.FreMLP_embed_size  
        self.hidden_size = configs.FreMLP_hidden_size 
        self.pre_length = configs.pred_len  
        self.channels = configs.enc_in  
        self.seq_length = configs.seq_len  
        self.type = configs.model_type


        self.sparsity_threshold = 0.01  
        self.scale = 0.02

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))


        # cof = math.floor((math.log(self.channels)**2 - 4 * math.log(self.channels) + 10) / 4)
        cof = math.ceil((math.sqrt(self.channels))/5)

        if self.type == 'mlp':
            self.Seasonal = nn.Sequential(
                nn.Linear(self.seq_length, 2*self.seq_length*cof),
                nn.Tanh(),
                nn.Dropout(0.4),
                nn.Linear(2*self.seq_length*cof, self.pre_length)
            )
            # self.Seasonal = nn.Sequential(
            #     nn.Linear(self.seq_length, 4096),
            #     nn.Tanh(),
            #     nn.Dropout(0.4),
            #     nn.Linear(4096, self.pre_length)
            # )

        elif self.type == 'linear':
            self.Seasonal = nn.Sequential(
                nn.Linear(self.seq_length, self.pre_length)
            )
        else:
            raise ValueError(f"Invalid model type: {self.type}")

        if self.type == 'mlp':
            self.fc = nn.Sequential(
                nn.Linear(self.seq_length * self.embed_size, self.hidden_size*cof),
                nn.LeakyReLU(),
                nn.Dropout(0.4),
                nn.Linear(self.hidden_size*cof, self.pre_length)
            )
           
            
        elif self.type == 'linear':
            self.fc = nn.Sequential(
                nn.Linear(self.seq_length * self.embed_size, self.pre_length)
            )
        else:
            raise ValueError(f"Invalid model type: {self.type}")    

        self.alpha = nn.Parameter(torch.zeros(self.channels))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embeddings, mean=0, std=0.02)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # for layer in self.fusion_mlp:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight)
        #         if layer.bias is not None:
        #             nn.init.zeros_(layer.bias)

    def tokenEmb(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        return x * self.embeddings


    def FreMLP(self, B, nd, dimension, x, r1, i1, rb1, ib1):
        o1_real = F.tanh(torch.einsum('bijd,dd->bijd', x.real, r1) - \
            torch.einsum('bijd,dd->bijd', x.imag, i1) + \
            rb1)


        o1_imag = F.tanh(torch.einsum('bijd,dd->bijd', x.imag, r1) + \
            torch.einsum('bijd,dd->bijd', x.real, i1) + \
            ib1)


        stacked = torch.stack([o1_real, o1_imag], dim=-1)

        y_shrink = F.softshrink(stacked, lambd=self.sparsity_threshold)  # self.sparsity_threshold, adaptive_threshold

        y = torch.view_as_complex(y_shrink)

        return y

    def MLP_temporal(self, x, B, N, L):

        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")

        return x



    def forward(self, x):

        B, T, N = x.shape


        x1 = self.tokenEmb(x)
        x1 = self.MLP_temporal(x1, B, N, T)  
        x1 = self.fc(x1.reshape(B, N, -1).contiguous()).permute(0, 2, 1)   
        x2 = self.Seasonal(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        alpha = self.alpha.view(1, 1, -1).contiguous()  # [1, 1, N]
        out = x1 + alpha * x2

        return out

