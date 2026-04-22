"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com), Biaolin Wen (robinbg@foxmail.com)

-----------------------------

model.py: Network Modules

(1) Encoder
(2) Recovery
(3) Generator
(4) Supervisor
(5) Discriminator
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
      for name,param in m.named_parameters():
        if 'weight_ih' in name:
          init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
          init.orthogonal_(param.data)
        elif 'bias' in name:
          param.data.fill_(0)


class EncoderRNN(nn.Module):
    """Embedding network between original feature space to latent space.

        Args:
          - input: input time-series features. (L, N, X) = (24, ?, 6)
          - h3: (num_layers, N, H). [3, ?, 24]

        Returns:
          - H: embeddings
        """
    def __init__(self, opt):
        super().__init__()
        self.rnn = nn.GRU(input_size=opt.z_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer)
       # self.norm = nn.BatchNorm1d(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        e_outputs, _ = self.rnn(input)
        H = self.fc(e_outputs)
        if sigmoid:
            H = self.sigmoid(H)
        return H
    

    '''
    Positional Enconding to enforce time order in Transformer model
    '''

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=500):
      super().__init__()

      pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
      position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

      div_term = torch.exp(
          torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
      )

      pe[:, 0::2] = torch.sin(position * div_term)
      if d_model % 2 == 0:
          pe[:, 1::2] = torch.cos(position * div_term)
      else:
          pe[:, 1::2] = torch.cos(position * div_term[:-1])

      pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
      self.register_buffer("pe", pe)

  def forward(self, x):
      # x: (L, N, d_model)
      L = x.size(0)
      return x + self.pe[:L]
    

class EncoderTransformer(nn.Module):
    """
    Embedding network between original feature space and latent space.

    Args:
      - input: input time-series features. (L, N, z_dim)

    Returns:
      - H: embeddings. (L, N, hidden_dim)
    """
    def __init__(self, opt):
        super().__init__()

        self.input_proj = nn.Linear(opt.z_dim, opt.d_model)
        self.pos_encoder = PositionalEncoding(opt.d_model, max_len=opt.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=opt.d_model,
            nhead=opt.nhead,
            dim_feedforward=opt.dim_feedforward,
            dropout=opt.dropout,
            batch_first=False
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=opt.num_layer
        )

        self.fc = nn.Linear(opt.d_model, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()

        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        # input: (L, N, z_dim)
        x = self.input_proj(input)       # (L, N, d_model)
        x = self.pos_encoder(x)          # (L, N, d_model)
        e_outputs = self.transformer(x)  # (L, N, d_model)
        H = self.fc(e_outputs)           # (L, N, hidden_dim)

        if sigmoid:
            H = self.sigmoid(H)

        return H


class Recovery(nn.Module):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - X_tilde: recovered data
    """
    def __init__(self, opt):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(input_size=opt.hidden_dim, hidden_size=opt.z_dim, num_layers=opt.num_layer)
        
      #  self.norm = nn.BatchNorm1d(opt.z_dim)
        self.fc = nn.Linear(opt.z_dim, opt.z_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        r_outputs, _ = self.rnn(input)
        X_tilde = self.fc(r_outputs)
        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)
        return X_tilde


class Generator(nn.Module):
    """Generator function: Generate time-series data in latent space.

    Args:
      - Z: random variables
      - T: input time information

    Returns:
      - E: generated embedding
    """
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(input_size=opt.z_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer)
     #   self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        g_outputs, _ = self.rnn(input)
      #  g_outputs = self.norm(g_outputs)
        E = self.fc(g_outputs)
        if sigmoid:
            E = self.sigmoid(E)
        return E
  
class Supervisor(nn.Module):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(input_size=opt.hidden_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer)
      #  self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        s_outputs, _ = self.rnn(input)
      #  s_outputs = self.norm(s_outputs)
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S


class DiscriminatorRNN(nn.Module):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """
    def __init__(self, opt):
        super().__init__()
        self.rnn = nn.GRU(input_size=opt.hidden_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer)
      #  self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        d_outputs, _ = self.rnn(input)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat
    

'''
Unimplemented Discriminator Transformer So far
'''
    
class DiscriminatorTransformer(nn.Module):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """
    def __init__(self, opt):
        super().__init__()
        self.rnn = nn.GRU(input_size=opt.hidden_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer)
      #  self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        d_outputs, _ = self.rnn(input)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat

