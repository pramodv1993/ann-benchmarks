import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch as T

#Generic Neural Network that can be used by Actor and Critic Networks
class NN(nn.Module):
  def __init__(self, learning_rate, inp_dims, layer1_dims, layer2_dims, out_dims):
    super(NN, self).__init__()
    self.learning_rate = learning_rate
    self.inp_dims = inp_dims
    self.fc1_dims = layer1_dims
    self.fc2_dims = layer2_dims
    self.out_dims = out_dims
    #layers
    self.inp_layer = nn.Linear(*self.inp_dims, self.fc1_dims)
    self.fc1 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.out_layer = nn.Linear(self.fc2_dims, self.out_dims)
    self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, X):
    X = T.tensor(X, dtype=T.float).to(self.device)
    X = F.relu(self.inp_layer(X))
    X = F.relu(self.fc1(X))
    X = self.out_layer(X)
    return X

class Agent(object):
  '''
  action : offsets - increase/decrease mu OR increase/decrease covariance vector(uniq elements)
  state : [inp_point, updated_mu, updated_covar, rank]
  rewards : based on proportion of FPS (or TPs?) in the jitters
  '''
  def __init__(self, alpha, beta, input_dims, layer1_dims,
               layer2_dims, n_actions, gamma = .99):
    self.actor = NN(alpha, input_dims, layer1_dims, layer2_dims, n_actions)
    self.critic = NN(beta, input_dims, layer1_dims, layer2_dims, out_dims=1 )
    self.gamma = gamma

  def choose_action(self, curr_state, num_dim):
    preds = self.actor.forward(curr_state).cpu().detach().numpy()
    return preds[:num_dim], preds[num_dim:]

  def learn(self, curr_state, reward, new_state, done):
    self.actor.optimizer.zero_grad()
    self.critic.optimizer.zero_grad()

    critic_val = self.critic.forward(curr_state)
    next_crtitic_val = self.critic.forward(new_state)
    reward = T.tensor(reward, dtype = T.float).to(self.actor.device)
    delta = reward + self.gamma * next_critic_val * (1- int(done)) - critic_val

    actor_loss = -self.log_prob * delta
    critic_loss = delta**2
    (actor_loss + critic_loss).backward()

    self.actor.optimizer.step()
    self.critic.optimizer.step()
