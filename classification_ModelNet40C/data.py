import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def load_data(data_path,corruption,severity, num_points):

    all_data_list = []
    all_label_list = []
    for corr in corruption:
        for sev in severity:
            DATA_DIR = os.path.join(data_path, 'data_' + corr + '_' +str(sev) + '.npy')
            all_data = np.load(DATA_DIR)[:,:num_points,:]
            all_data_list.append(all_data)

            LABEL_DIR = os.path.join(data_path, 'label.npy')
            all_label = np.load(LABEL_DIR)
            all_label_list.append(all_label)

    
    return np.concatenate(all_data_list, axis=0), np.concatenate(all_label_list, axis=0)

    
class ModelNet40C(Dataset):
    def __init__(self, data_path,corruption,severity, partition, num_points):
        self.corruption = corruption
        self.severity = severity
        self.data_path = data_path
        self.partition = partition
        self.data = None
        self.label = None
        self.num_points = num_points

        self.data_all, self.label_all = load_data(self.data_path, self.corruption, self.severity, self.num_points)
        np.random.seed(1)
        idx = np.arange(0, self.data_all.shape[0])
        np.random.shuffle(idx)
        if self.partition == "train":
            self.data = self.data_all[idx][:int(self.data_all.shape[0] * 0.7)]
            self.label = self.label_all[idx][:int(self.data_all.shape[0] * 0.7)]
        else:
            self.data = self.data_all[idx][int(self.data_all.shape[0] * 0.7):]
            self.label = self.label_all[idx][int(self.data_all.shape[0] * 0.7):]
        

    def __getitem__(self, item):
        pointcloud = self.data[item]#[:self.num_points]
        label = self.label[item]
        return pointcloud, label.item()

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    data_path = "/content/pointMLP-pytorch/classification_ModelNet40C/data/modelnet40_c"
    corruption = ["background", "cutout", "density", "density_inc", "distortion", "distortion_rbf", "distortion_rbf_inv", "gaussian", "impulse", "lidar", "occlusion", "rotation", "shear", "uniform", "upsampling"]
    severity = [1, 2, 3, 4, 5]
    data_train = ModelNet40C(data_path, corruption, severity, partition="train", num_points=649)
    data_test = ModelNet40C(data_path, corruption, severity, partition="test", num_points=649)
    print(data_train.data.shape)
    train_loader = DataLoader(data_train, num_workers=2,
                              batch_size=1024, shuffle=True, drop_last=True)
    test_loader = DataLoader(data_test, num_workers=2,
                              batch_size=1024, shuffle=False)

