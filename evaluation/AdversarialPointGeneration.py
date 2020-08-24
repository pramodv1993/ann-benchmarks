#train an agent that generates jitters for a point that is a FP

from eval import EvaluationService
from Agent import Agent
from SearchSpaceEnvironment import SearchSpaceEnvironment

import sys,os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from ann_benchmarks.algorithms import flann
import numpy as np
#prepare data
res_file = '/home/pramod/Desktop/knns/ann-benchmarks/results/glove-100-angular/10/flann/angular_0_95' #result from flann algorithm for the glove dataset with 100 dimensions
orig_file = '/home/pramod/Desktop/knns/ann-benchmarks/data/glove-100-angular.hdf5'
eval_service = EvaluationService(orig_file = orig_file, res_file = res_file)
fps = eval_service.fetch_fps_and_tps()[0]['FP']
print("Fetched false positives..")
flann = flann.FLANN(metric = 'angular', target_precision = .95)
#train agent
TEST_POINT = fps[0]
NUM_EPISODES  = 10
NUM_SAMPLE_POINTS = 1000
INIT_MU = np.random.normal(size = (TEST_POINT.shape[0], ))
num_uniq_ele = TEST_POINT.shape[0] * (TEST_POINT.shape[0] + 1) /2 #upper/lower traingle from the covariance matrix
INIT_COVAR = [np.random.rand() for _ in range(int(num_uniq_ele))]
NUM_STATES = 20
TARGET_ALGORITHM = flann
TARGET_ARGS = dict(metric = 'angular', target_precision = .95)
if __name__=="__main__":
  n_outputs = len(INIT_MU) + len(INIT_COVAR) #actor's output
  agent = Agent(alpha=.00005, beta=.00001,
                n_actions = n_outputs,
                input_dims=[len(TEST_POINT) + n_outputs +1],
                layer1_dims=256,
                layer2_dims = 256)

  env = SearchSpaceEnvironment(test_point = TEST_POINT,
                               k = 10,
                               num_sample_points = NUM_SAMPLE_POINTS,
                               init_mu = INIT_MU,
                               init_covar = INIT_COVAR,
                               num_states = NUM_STATES,
                               target_algorithm = TARGET_ALGORITHM,
                               target_args = TARGET_ARGS,
                               train_points = train_pts)
  exit(-1)
#   '''episode lasts until all the points sampled from a computed distribution are
#   adverserial examples'''
#   for i in range(NUM_EPISODES):
#     done = False
#     score = 0
#     curr_state = env.reset_search_environment()
#     while not done:
#       offset_mu, offset_covar  = agent.choose_action(curr_state, len(TEST_POINT))
#       new_state, reward, done = env.update_mean_and_covar(offset_mu, offset_covar)
#       #debug
#       break
#     break
#     #   score += reward
#
#
#
