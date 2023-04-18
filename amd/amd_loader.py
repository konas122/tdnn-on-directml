import os
import torch
import numpy as np
import multiprocessing
import amd.amd_audio as audio
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader

# transcript_filename = "..\\data\\transcript.txt"
# test_path = "..\\data\\test"
# train_path = "..\\data\\train"
# dev_path = "..\\data\\dev"


class MyDataset(Dataset):
    def __init__(self, data_dict=None, people_num=None, train=True, mel=True, noise=False):
        super(MyDataset, self).__init__()
        self.noise = noise
        self.mel = mel
        self.train = train
        self.data_dict = data_dict
        self.spect = []
        self.labels = []
        if data_dict is None or people_num is None:
            raise Exception(f'Error: data_dtc {data_dict} is empty\n')
        else:
            self.people_num = people_num
            self._preprocess()

    def _preprocess(self):
        out = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self._audio)(key) for key in self.data_dict)
        self.spect = [value for value, _ in out]
        self.labels = [value for _, value in out]
        # self.spect = torch.tensor([item.cpu().detach().numpy() for item in self.spect]).float()
        self.labels = torch.from_numpy(np.array(self.labels)).long()

    def _audio(self, key):
        spec = audio.loadWAV(filename=key)
        # return spec, self.data_dict[key], key
        return spec, self.data_dict[key]

    def __getitem__(self, item):
        label = self.labels[item]
        spec = self.spect[item]
        return spec, label

    def __len__(self):
        return len(self.labels)


def load_files(mode="train", folder_num=-1, file_num=-1, k=1.5):
    path = "..\\data"
    train, test = {}, {}
    if mode == "train":
        path = path + '\\train'
    elif mode == "test":
        path = path + '\\test'
    elif mode == "dev":
        path = path + "\\dev"
    else:
        raise Exception(f'Error: mode {mode} 不存在')
    dirs = os.listdir(path)

    if 0 < folder_num < len(dirs):
        if mode == "train":
            num = np.arange(folder_num)
        else:
            num = np.random.choice(len(dirs), folder_num, replace=False)
    else:
        num = np.arange(len(dirs))
        folder_num = len(dirs)
    if k <= 0 or k >= 9:
        k = 1.5

    count = 0
    folder_path = []
    for i in num:
        file_path = dirs[i]
        folder_path.append(file_path)
        file_path = os.path.join(path, file_path)
        tmp_files = os.listdir(file_path)
        sub_files = [tmp_files[file] for file in range(len(tmp_files))
                     if tmp_files[file][-4:] == ".wav"]

        if file_num > len(sub_files):
            file_num = len(sub_files)
        elif file_num < 10:
            file_num = 10
        np.random.shuffle(sub_files)
        train_num = int(file_num // (k + 1) * k + 1)
        # test_num = file_num - train_num

        for j in range(train_num):
            wav_file = os.path.join(file_path, sub_files[j])
            train[wav_file] = count
        for j in range(train_num, file_num):
            wav_file = os.path.join(file_path, sub_files[j])
            test[wav_file] = count
        count += 1
    return train, test, folder_num


if __name__ == '__main__':
    Reverse = False
    train_dict, test_dict, number = load_files("train", 40, 20, 1.5)
    train_dataset = MyDataset(train_dict, number, True, True, False)
    test_dataset = MyDataset(test_dict, number, False, True, False)
    print(len(train_dataset), len(test_dataset))
    train_iter = DataLoader(dataset=train_dataset, batch_size=6, shuffle=True, drop_last=True, num_workers=4)
    print(len(train_iter))
    a = None
    for b, (x, y) in enumerate(train_iter):
        if b == 0:
            a = x
        print(x.shape, y)
    print(a[0].shape)
