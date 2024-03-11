import torch
import torch.nn as nn
import numpy as np

class Critic(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        # set hyperparams
        self.latent_dim = hparams["latent_dim"] 
        self.hparams = hparams
        self.convolution = nn.Sequential(
          #N,1,28,28
          #can add dropout later
          nn.Conv2d(3,self.hparams["dc1_num"], self.hparams["dc1_size"],self.hparams["dc1_stride"]),
          nn.BatchNorm2d(self.hparams["dc1_num"]),
          nn.LeakyReLU(),
          nn.MaxPool2d(self.hparams[ "dc1p_size"],self.hparams[ "dc1p_stride"]),
          nn.Conv2d(self.hparams["dc1_num"],self.hparams["dc2_num"], self.hparams["dc2_size"],self.hparams["dc2_stride"]),
          nn.BatchNorm2d(self.hparams["dc2_num"]),
          nn.LeakyReLU(),
          nn.AvgPool2d(self.hparams[ "dc2p_size"],self.hparams[ "dc2p_stride"]),
          #nn.Conv2d(self.hparams["dc2_num"],self.hparams["dc3_num"], self.hparams["dc3_size"],self.hparams["dc3_stride"]),
          #nn.BatchNorm2d(self.hparams["dc3_num"]),
          #nn.ReLU(),
          #nn.MaxPool2d(self.hparams[ "dc3p_size"],self.hparams[ "dc3p_stride"]),
          #nn.Conv2d(self.hparams["dc3_num"],self.hparams["dc4_num"], self.hparams["dc4_size"],self.hparams["dc4_stride"]),
          #nn.BatchNorm2d(self.hparams["dc4_num"]),
          #nn.AvgPool2d(self.hparams[ "dc4p_size"],self.hparams[ "dc4p_stride"]),
          #nn.LeakyReLU(),
        )
        self.conv_strided = nn.Sequential(
          #N,1,28,28
          #can add dropout later
          nn.Conv2d(3,self.hparams["dc1_num"], self.hparams["dc1_size"],self.hparams["dc1_stride"]),
          nn.LeakyReLU(),
          nn.BatchNorm2d(self.hparams["dc1_num"]),
          nn.Conv2d(self.hparams["dc1_num"],self.hparams["dc2_num"], self.hparams["dc2_size"],self.hparams["dc2_stride"]),
          nn.LeakyReLU(),
          nn.BatchNorm2d(self.hparams["dc2_num"]),
          nn.Conv2d(self.hparams["dc2_num"],self.hparams["dc3_num"], self.hparams["dc3_size"],self.hparams["dc3_stride"]),
          nn.LeakyReLU(),
          nn.BatchNorm2d(self.hparams["dc3_num"]),
          nn.Conv2d(self.hparams["dc3_num"],self.hparams["dc4_num"], self.hparams["dc4_size"],self.hparams["dc4_stride"]),
          nn.LeakyReLU(),
          nn.BatchNorm2d(self.hparams["dc4_num"]),
          nn.Conv2d(self.hparams["dc4_num"],self.hparams["dc5_num"], self.hparams["dc5_size"],self.hparams["dc4_stride"]),
          nn.BatchNorm2d(self.hparams["dc5_num"]),
          nn.LeakyReLU()
          
        )

        self.fcnn = nn.Sequential(
          nn.Linear(self.hparams["dc5_num"], self.hparams["dl1"]),
          nn.BatchNorm1d(self.hparams["dl1"]),
          nn.LeakyReLU(),
          nn.Linear(self.hparams["dl1"], self.hparams["dl2"]),
          nn.BatchNorm1d(self.hparams["dl2"]),
          nn.LeakyReLU(),
          nn.Linear(self.hparams["dl2"], self.hparams["dl3"]),
          nn.BatchNorm1d(self.hparams["dl3"]),
          nn.LeakyReLU(),
          nn.Linear(self.hparams["dl3"], 1),
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        x = self.conv_strided(x)
        x = x.view(-1, self.hparams["dc5_num"]) #use dc4 if using convolution
        x = self.fcnn(x)
        return x

class Generator(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.latent_dim = hparams["latent_dim"]

        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################
        self.upsampling = nn.Sequential(

          nn.Linear(self.latent_dim, hparams["gl1"]),
          nn.BatchNorm1d(self.hparams["gl1"]),
          nn.LeakyReLU(),
          nn.Linear(hparams["gl1"], hparams["gl2"]),
          nn.BatchNorm1d(self.hparams["gl2"]),
          nn.LeakyReLU(),
          nn.Linear(hparams["gl2"], hparams["gl3"]),
          nn.BatchNorm1d(self.hparams["gl3"]),
          nn.LeakyReLU(),
          nn.Linear(hparams["gl3"], hparams["gl4"]),
          nn.BatchNorm1d(self.hparams["gl4"]),
          nn.LeakyReLU()
        )

        self.upconvolution = nn.Sequential(
          nn.ConvTranspose2d(self.latent_dim,self.hparams["gc1_out_channels"], self.hparams["gc1_size"],self.hparams["gc1_stride"]),
          nn.ReLU(),
          nn.BatchNorm2d(self.hparams["gc1_out_channels"]),
          nn.ConvTranspose2d(self.hparams["gc1_out_channels"],self.hparams["gc2_out_channels"], self.hparams["gc2_size"],self.hparams["gc2_stride"]),
          nn.ReLU(),
          nn.BatchNorm2d(self.hparams["gc2_out_channels"]),
          nn.ConvTranspose2d(self.hparams["gc2_out_channels"],self.hparams["gc3_out_channels"], self.hparams["gc3_size"],self.hparams["gc3_stride"]),
          nn.Tanh(),
        )
        """
          nn.ReLU(),
          nn.BatchNorm2d(self.hparams["gc3_out_channels"]),
          nn.ConvTranspose2d(self.hparams["gc3_out_channels"],self.hparams["gc4_out_channels"], self.hparams["gc4_size"],self.hparams["gc4_stride"]),
          nn.ReLU(),
          nn.BatchNorm2d(self.hparams["gc4_out_channels"]),
          nn.ConvTranspose2d(self.hparams["gc4_out_channels"],self.hparams["gc5_out_channels"], self.hparams["gc5_size"],self.hparams["gc5_stride"]),
          nn.ReLU(),
          nn.BatchNorm2d(self.hparams["gc5_out_channels"]),
          nn.ConvTranspose2d(self.hparams["gc5_out_channels"],self.hparams["gc6_out_channels"], self.hparams["gc6_size"],self.hparams["gc6_stride"]),
          """
        self.mlp512 = nn.Sequential(
          nn.Linear(self.latent_dim, 512),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.Linear(512, 512),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.Linear(512, 512),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.Linear(512, 512),
          nn.BatchNorm1d(512),
          nn.LeakyReLU(),
          nn.Linear(512,784),
          nn.Tanh()
        )

        self.cnn = nn.Sequential(
          nn.Conv2d(1,self.hparams["gc1_num"], self.hparams["gc1_size"],self.hparams["gc1_stride"]),
          nn.BatchNorm2d(self.hparams["gc1_num"]),
          nn.ReLU(),
          nn.MaxPool2d(self.hparams[ "gc1p_size"],self.hparams[ "gc1p_stride"]),
          nn.Conv2d(self.hparams["gc1_num"],self.hparams["gc2_num"], self.hparams["gc2_size"],self.hparams["gc2_stride"]),
          nn.BatchNorm2d(self.hparams["gc2_num"]),
          nn.MaxPool2d(self.hparams[ "gc2p_size"],self.hparams[ "gc2p_stride"]),
          nn.Conv2d(self.hparams["gc2_num"],self.hparams["gc3_num"], self.hparams["gc3_size"],self.hparams["gc3_stride"]),
          nn.BatchNorm2d(self.hparams["gc3_num"]),
          nn.MaxPool2d(self.hparams[ "gc3p_size"],self.hparams[ "gc3p_stride"]),
          nn.Conv2d(self.hparams["gc3_num"],self.hparams["gc4_num"], self.hparams["gc4_size"],self.hparams["gc4_stride"]),
          nn.BatchNorm2d(self.hparams["gc4_num"]),
          nn.Tanh()
          
          #end in N,1,28,28
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        #x = self.upsampling(x)
        #x = x.view(-1,1,self.hparams["upscale_to"],self.hparams["upscale_to"])
        #x = self.cnn(x)
        #return x

        #x = self.mlp512(x)
        #x = x.view(-1,1,28,28)
        #return x

        #add an 1 dimensional layer in the middle
        x = x.view(-1,self.latent_dim,1, 1)
        x = self.upconvolution(x)
        return x

class WeightClipper(object):
    
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-0.02,0.021)
            module.weight.data = w



