import h5py
import fire
import os
class EvaluationService:
        def __init__(self, orig_file = None, res_file = None):
            self.orig_file = orig_file
            self.res_file_path = res_file

        def _fetch_fps_in_file(self, res_file):
            if self.orig_file is None:
                raise Exception("Please specify ground truth file..")
            fps = list()
            orig = h5py.File(self.orig_file, 'r')
            res = h5py.File(res_file,'r')
            res_neighbors = res['neighbors']
            test_size, top_k = res_neighbors.shape
            #selecting only top k neighbors from ground truth
            true_neighbors = orig['neighbors'][:,:top_k]
            for i in range(test_size):
                neighbors_match = all(true==res for (true, res)
                            in zip(true_neighbors[i], res_neighbors[i]))
                if not neighbors_match:
                    fps.append(res_neighbors[i,:])
            print("Length of neighbors that dont match with true neighbors {}/{}".format(len(fps), test_size))

        def fetch_fps(self):
            if os.path.isdir(self.res_file_path):
                print("Reading files from path..")
                for file in os.listdir(self.res_file_path):
                    print("Matching neighbors from file {}".format(file))
                    self._fetch_fps_in_file(os.path.join(self.res_file_path, file))
            else:
                print("Reading file..")
                self._fetch_fps_in_file(self.res_file_path)


if __name__=="__main__":
	fire.Fire(EvaluationService)
