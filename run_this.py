import os
import torch
import numpy as np
from amd.amd_eval import try_gpu
from amd.amd_pretrain import Pretrain_TDNN
from amd.amd_tools import dic_process, get_embedding


def load_files(mode="train", folder_num=-1, file_num=-1, k=1.5):
    path = ".\\data"
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


if __name__ == "__main__":
    Device = try_gpu()
    model = Pretrain_TDNN(420, 1024, output_embedding=False, not_grad=True)
    model.load_parameters('param.model')
    model.eval()
    model.to(Device)

    labels = []
    embed_dict = {}
    score_list = []
    enroll, _, people_num = load_files("test", 10, 10)  # 后两个数字：读取文件夹的个数；每个文件夹中要读取音频的个数
    enroll = dic_process(enroll)

    for key in enroll:
        count = 0
        embed = None
        for name in enroll[key]:
            if count >= len(enroll[key]):
                break
            count += 1
            embedding = get_embedding(model, name, Device)

            if count == 1:
                embed = embedding
            else:
                embed = torch.cat([embed, embedding])
        embed = torch.mean(embed, dim=0).unsqueeze(0)
        embed_dict[key] = embed

    num = int(input("\n\n输入测试人数："))
    for i in range(num):  # 测试次数
        mark = None
        max_score = 0
        filepath = input("输入音频路径：")  # 测试音频的路径
        embedding = get_embedding(model, filepath, Device)
        for key in embed_dict.keys():
            embed1 = embed_dict[key]
            score = torch.matmul(embedding, embed1.mT).cpu().numpy()
            if score > max_score:
                max_score = score
                mark = key
        if max_score < 0.9415:
            print('该音频不在库中', max_score[0][0])
        else:
            print(mark, max_score[0][0])
