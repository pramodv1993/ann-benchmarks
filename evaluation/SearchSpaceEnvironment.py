#environment that evaluates the approximate KNNS algorithms for a given test point
import cudf
from cuml.neighbors import NearestNeighbors as cuml_NN
from sklearn.neighbors import NearestNeighbors as sklearn_NN
import numpy as np

class SearchSpaceEnvironment:
  def __init__(self, test_point, k, num_sample_points, init_mu, init_covar,
               num_states, target_algorithm, train_points):
    self.test_point = test_point
    self.dim = len(test_point)
    self.done = False
    self.k = k
    self.num_sample_points = num_sample_points
    self.init_mu = init_mu
    self.init_covar = init_covar
    self.curr_mean = self.init_mu
    self.curr_covar = self.init_covar
    self.num_states = num_states
    self.target_algorithm = target_algorithm
    self.bucket_size = self.num_sample_points/ self.num_states
    self.curr_state = np.concatenate((self.test_point,
                                      self.init_mu,
                                      self.init_covar, [0]))
    print("Fitting the training points..")
    self.true_nn = cuml_NN()
    self.true_nn.fit(train_points)

    if len(self.init_covar) != self.dim *(self.dim +1)/2:
      raise Exception('Insufficient unique elements specified for covariance matrix')


  def _construct_covar_mat(self, uniq_vals, dim):
    res = np.zeros((dim,dim))
    diag_eles, covar_eles = uniq_vals[:dim], uniq_vals[dim:]
    res[np.triu_indices_from(res, k = 1)] = covar_eles
    res = res + res.T
    res[np.diag_indices_from(res)] = diag_eles
    return res

  def sample_points_from_space(self, mean, covar, sample_size):
    full_covar_matrix = self._construct_covar_mat(covar, self.dim)
    return np.random.multivariate_normal(mean, full_covar_matrix, sample_size)

  def evaluate_points(self, sampled_points, **kwargs):
    true_distances, true_neighbors_indices = self.true_nn.kneighbors(sampled_points, self.k)#slow
    approx_neighbors = self.target_algorithm(sampled_points, self.k, **kwargs)#slow
    num_fps = 0
    for true_neighbors_of_point, approx_neighbors_of_point in zip(true_neighbors, approx_neighbors):
      #order of true neighbors has to retained
      #TODO: ignore order of the neighbors
      if true_neighbors_of_point != approx_neighbors_of_point:
        num_fps += 1
    return num_fps


  #equivalent of env.step
  def update_mean_and_covar(self, offset_mu = None, offset_covar = None):
    '''
    - change mean and covar
    - sample points from the modified space
    - evaluate the true and approximate neighbors of the sampled points
    '''
    self.curr_mean +=  offset_mu
    self.curr_covar += offset_covar
    print("Sampling points from new space..")
    sampled_points = self.sample_points_from_space(self.curr_mean, self.curr_covar, self.num_sample_points)


    #@TODO fill below methods
    num_fps = self.evaluate_points(sampled_points)
    #if lesser proportion of jitters are FPs, rewards are higher
    reward = 100 * np.log(num_fps/self.num_sample_points)
    if num_fps == self.num_sample_points:
      self.done = True
    else:
      #state: (point, mean, covar's uniq elements, rank)
      self.curr_state = np.concatenate((self.test_point,
                                      self.init_mu,
                                      self.init_covar, [self.num_states - (self.num_fps // self.bucket_size)]))
    return self.curr_state, reward, self.done
    return sampled_points

  def reset_search_environment(self):
    self.done = False
    self.curr_state = np.concatenate((self.test_point,
                                      self.init_mu,
                                      self.init_covar, [0]))
    self.curr_mean = self.init_mu
    self.curr_covar = self.init_covar
    return self.curr_state
