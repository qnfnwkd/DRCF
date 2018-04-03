from __future__ import unicode_literals, print_function, division
import torch
from torch.autograd import Variable
import torch.nn as nn

class DRCF(nn.Module):
	def __init__(self, embedding_dim, rnn_steps, user_num, venue_num, samples_num):
		super(DRCF, self).__init__()

		## hyper-parameters
		self.embedding_dim = embedding_dim
		self.rnn_steps = rnn_steps

		## parameters
		self.user_num = user_num
		self.venue_num = venue_num
		self.samples_num = samples_num

		## embedding layers
		# user embedding
		self.p_g_d = nn.Embedding(self.user_num, self.embedding_dim)
		self.p_m_d = nn.Embedding(self.user_num, self.embedding_dim)
		self.p_g = nn.Embedding(self.user_num, self.embedding_dim)
		self.p_m = nn.Embedding(self.user_num, self.embedding_dim)

		# venue embedding
		self.q_g = nn.Embedding(self.venue_num, self.embedding_dim)
		self.q_m = nn.Embedding(self.venue_num, self.embedding_dim)
		self.q_g_d = nn.Embedding(self.venue_num, self.embedding_dim)
		self.q_m_d = nn.Embedding(self.venue_num, self.embedding_dim)

		## RNN layers
		self.d_g_u = nn.LSTM(self.embedding_dim, self.embedding_dim, batch_first=True)
		self.d_m_u = nn.LSTM(self.embedding_dim, self.embedding_dim, batch_first=True)
		self.d_gd_u = nn.LSTM(self.embedding_dim, self.embedding_dim, batch_first=True)
		self.d_md_u = nn.LSTM(self.embedding_dim, self.embedding_dim, batch_first=True)

		## MLRP layers
		self.mlp_1 = nn.Linear(3*self.embedding_dim, 2*self.embedding_dim)
		self.mlp_2 = nn.Linear(2*self.embedding_dim, self.embedding_dim)
		self.mlp_3 = nn.Linear(self.embedding_dim, self.embedding_dim)

		## DRMF layer
		self.drmf = nn.Linear(2*self.embedding_dim + 2, 1)

	def forward(self, user, candidate, checkins, samples):
		# User Latent Factor, Shape: (batch, embedding_dim)
		p_g_d = self.p_g_d(user)
		p_m_d = self.p_m_d(user)
		p_g = self.p_g(user)
		p_m = self.p_m(user)

		# Venue Latent Factor
		## Positive Candidate, Shape: (batch, embedding_dim)
		q_g = self.q_g(candidate)
		q_m = self.q_m(candidate)
		q_g_d = self.q_g_d(candidate)
		q_m_d = self.q_m_d(candidate)

		## Negative Samples, Shape: (batch, samples_num, embedding_dim)
		neg_q_g = self.q_g(samples)
		neg_q_m = self.q_m(samples)
		neg_q_g_d = self.q_g_d(samples)
		neg_q_m_d = self.q_m_d(samples)

		self.d_g_u.flatten_parameters()
		self.d_m_u.flatten_parameters()
		self.d_gd_u.flatten_parameters()
		self.d_md_u.flatten_parameters()

		# Dyanamic User Latent Factor (RNN)
		_, d_g_u = self.d_g_u(self.q_g(checkins))
		d_g_u = d_g_u[0].squeeze()

		_, d_m_u = self.d_m_u(self.q_m(checkins))
		d_m_u = d_m_u[0].squeeze()

		_, d_gd_u = self.d_gd_u(self.q_g_d(checkins))
		d_gd_u = d_gd_u[0].squeeze()

		_, d_md_u = self.d_md_u(self.q_m_d(checkins))
		d_md_u = d_md_u[0].squeeze()

		# GRMF layer
		_grmf = d_g_u*p_g

		## Positive
		grmf = _grmf*q_g
		grmf = d_g_u*p_g*q_g # Shape: (batch, embedding_dim)
		
		## Negative
		neg_grmf = _grmf.unsqueeze(1).expand(-1, self.samples_num, self.embedding_dim) * neg_q_g

		# MLRP layer
		_mlrp = torch.cat([d_m_u, p_m], 1) # Shape: (batch, 2*embedding_dim)

		## Positive
		mlrp = torch.cat([_mlrp, q_m], 1)
		mlrp = self.mlp_1(mlrp)
		mlrp = self.mlp_2(mlrp)
		mlrp = self.mlp_3(mlrp)

		## Negative Shape: (batch, samples_num, 3*embedding_dim)
		neg_mlrp = torch.cat([_mlrp.unsqueeze(1).expand(-1, self.samples_num, 2*self.embedding_dim), neg_q_m], 2)
		neg_mlrp = self.mlp_1(neg_mlrp)
		neg_mlrp = self.mlp_2(neg_mlrp)
		neg_mlrp = self.mlp_3(neg_mlrp)

		# RMF layer
		_rmf_left = (d_gd_u + p_g_d)
		_rmf_right = (d_md_u + p_m_d)

		## Positive
		rmf = torch.cat([(_rmf_left*q_g_d).sum(1, keepdim=True), (_rmf_right*q_m_d).sum(1, keepdim=True)], 1)

		## Negative
		neg_rmf = torch.cat([(_rmf_left.unsqueeze(1).expand(-1, self.samples_num, self.embedding_dim)*neg_q_g_d).sum(2, keepdim=True), (_rmf_right.unsqueeze(1).expand(-1, self.samples_num, self.embedding_dim)*neg_q_m_d).sum(2, keepdim=True)], 2)

		# output layer
		## Positive Shape: (batch, 1)
		output = self.drmf(torch.cat([grmf, mlrp, rmf], 1))

		## Negative Shape: (batch, samples_num, 1)
		neg_output = self.drmf(torch.cat([neg_grmf, neg_mlrp, neg_rmf], 2))

		return output.unsqueeze(1).expand(-1, self.samples_num, 1) - neg_output

	def evaluation(self, user, checkins):
		# User Latent Factor, Shape: (batch, embedding_dim)
		p_g_d = self.p_g_d(user)
		p_m_d = self.p_m_d(user)
		p_g = self.p_g(user)
		p_m = self.p_m(user)

		# Dyanamic User Latent Factor (RNN)

		self.d_g_u.flatten_parameters()
		self.d_m_u.flatten_parameters()
		self.d_gd_u.flatten_parameters()
		self.d_md_u.flatten_parameters()

		_, d_g_u = self.d_g_u(self.q_g(checkins))
		d_g_u = d_g_u[0].squeeze()

		_, d_m_u = self.d_m_u(self.q_m(checkins))
		d_m_u = d_m_u[0].squeeze()

		_, d_gd_u = self.d_gd_u(self.q_g_d(checkins))
		d_gd_u = d_gd_u[0].squeeze()

		_, d_md_u = self.d_md_u(self.q_m_d(checkins))
		d_md_u = d_md_u[0].squeeze()

		# GRMF layer
		grmf = (d_g_u*p_g).unsqueeze(1).expand(-1, self.venue_num, self.embedding_dim)
		grmf = grmf*self.q_g.weight # Shape: (batch, venue_num, embedding_dim)
		
		# MLRP layer
		mlrp = torch.cat([d_m_u, p_m], 1).unsqueeze(1).expand(-1, self.venue_num, 2*self.embedding_dim) # Shape: (batch, 2*embedding_dim)
		mlrp = torch.cat([mlrp, self.q_m.weight.unsqueeze(0).expand(mlrp.size(0), self.venue_num, self.embedding_dim)], 2)
		mlrp = self.mlp_1(mlrp)
		mlrp = self.mlp_2(mlrp)
		mlrp = self.mlp_3(mlrp)

		# RMF layer
		rmf_left = (d_gd_u + p_g_d).unsqueeze(1).expand(-1, self.venue_num, self.embedding_dim)
		rmf_right = (d_md_u + p_m_d).unsqueeze(1).expand(-1, self.venue_num, self.embedding_dim)
		rmf = torch.cat([(rmf_left*self.q_g_d.weight).sum(2, keepdim=True), (rmf_right*self.q_m_d.weight).sum(2, keepdim=True)], 2)

		# output layer
		output = self.drmf(torch.cat([grmf, mlrp, rmf], 2)).squeeze()
		_, rank = torch.sort(output, descending=True)

		return rank
