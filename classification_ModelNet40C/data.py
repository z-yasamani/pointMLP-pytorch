import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def load_data(data_path,corruption,severity):

    all_data_list = []
    for corr in corruption:
        for sev in severity:
            DATA_DIR = os.path.join(data_path, 'data_' + corr + '_' +str(sev) + '.npy')
            all_data = np.load(DATA_DIR)
            all_data_list.append(all_data)
    
    LABEL_DIR = os.path.join(data_path, 'label.npy')
    all_label = np.load(LABEL_DIR)

    return np.concatenate(all_data_list), all_label

    
class ModelNet40C(Dataset):
    def __init__(self, data_path,corruption,severity, partition):
        self.corruption = corruption
        self.severity = severity
        self.data_path = data_path
        self.partition = partition
        all_data = False
        if len(self.corruption) and len(self.severity) != 1:
            all_data = True

        self.data, self.label = load_data(self.data_path, self.corruption, self.severity, all_data)
        np.random.seed(1)
        idx = np.random.shuffle(np.arange(0, self.data.shape[0]))
        if partition == "train":
            self.data = self.data[idx][:int(self.data.shape[0] * 0.7)]
            self.label = self.label[idx][:int(self.data.shape[0] * 0.7)]
        else:
            self.data = self.data[idx][int(self.data.shape[0] * 0.7):]
            self.label = self.label[idx][int(self.data.shape[0] * 0.7):]
        # self.num_points = num_points

    def __getitem__(self, item):
        pointcloud = self.data[item]#[:self.num_points]
        label = self.label[item]
        return pointcloud, label.item()

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    data_path = ""
    corruption = ["background", "cutout", "density", "density_inc", "distortion", "distortion_rbf", "distortion_rbf_inv", "gaussian", "impulse", "lidar", "occlusion", "rotation", "shear", "uniform", "upsampling"]
    severity = [1, 2, 3, 4, 5]
    data_train = ModelNet40C(data_path, corruption, severity, partition="train")
    data_test = ModelNet40C(data_path, corruption, severity, partition="test")
    train_loader = DataLoader(data_train, num_workers=4,
                              batch_size=2, shuffle=True, drop_last=True)
    test_loader = DataLoader(data_test, num_workers=4,
                              batch_size=2, shuffle=False)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")

