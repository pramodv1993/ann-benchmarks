import h5py
import fire
import os
import time
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class EvaluationService:
        def __init__(self, orig_file = None, res_file = None):
            self.orig_file = orig_file
            self.res_file_path = res_file

        def _fetch_fps_and_tps_in_file(self, res_file):
            if self.orig_file is None:
                raise Exception("Please specify ground truth file..")
            fps = list()
            tps = list()
            result = dict()
            flags = list()
            orig = h5py.File(self.orig_file, 'r')
            res = h5py.File(res_file,'r')
            res_neighbors = res['neighbors']
            test_size, top_k = res_neighbors.shape
            #selecting only top k neighbors from ground truth
            true_neighbors = orig['neighbors'][:,:top_k]
            query_points = orig['test']
            for i in range(test_size):
                ## TODO: try not doing a 1-1 match of the neighbors, but just checking if its exists in the list of the true neighbors
                all_neighbors_match = all(true==res for (true, res)
                            in zip(true_neighbors[i], res_neighbors[i]))
                if not all_neighbors_match:
                    flags.append('FP')
                    fps.append(query_points[i])
                else:
                    tps.append(query_points[i])
                    flags.append('TP')
            print("Length of neighbors that dont match with true neighbors {}/{}".format(len(fps), test_size))
            print("Length of neighbors that  match with the true neighbors {}/{}".format(len(tps), test_size))
            result['FP'] = fps
            result['TP'] = tps
            result['FLAGS'] = flags
            return result

        def fetch_fps_and_tps(self):
            results = list()
            if os.path.isdir(self.res_file_path):
                print("Reading files from path..")
                for file in os.listdir(self.res_file_path):
                    print("Matching neighbors from file {}".format(file))
                    results.append(self._fetch_fps_and_tps_in_file(os.path.join(self.res_file_path, file)))
            else:
                print("Reading file..")
                results.append(self._fetch_fps_and_tps_in_file(self.res_file_path))
            return results

        def dim_reduction(self, method):
            orig = h5py.File(self.orig_file, 'r')
            query_points = orig['test']
            print(query_points.shape)
            results = self.fetch_fps_and_tps()
            color = {'FP':'red', 'TP': 'green'}
            st = time.time()
            print("Subjecting test points to dimensionality reduction - {}".format(method))
            if method == "tsne":
                reduced_points = TSNE(n_components = 2).fit_transform(query_points)
            elif method == "pca":
                reduced_points = PCA(n_components = 2).fit_transform(query_points)
            print("Time taken : {}".format(time.time() - st))


            for i, result in enumerate(results):
                print("Visualising FPs and TPs..")
                plt_color = [color[flag] for flag in result['FLAGS']]
                plt.scatter(reduced_points[:,0], reduced_points[:,1], color = plt_color, alpha = .7)
                plt.show()


if __name__== "__main__":
	fire.Fire(EvaluationService)
