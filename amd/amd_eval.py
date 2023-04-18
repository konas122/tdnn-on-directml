import torch
import torch_directml
import amd.amd_tools as tools
# from d2l import torch as d2l
from amd.amd_pretrain import Pretrain_TDNN


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch_directml.device()


if __name__ == "__main__":
    people_num, data_per_people = 10, 10

    Device = try_gpu()

    model2 = Pretrain_TDNN(420, 1024, output_embedding=False, not_grad=False)
    model2.load_parameters('../param.model')

    tools.eval_net(model2, Device, people_num, data_per_people)

    # model2.save_parameters('param2.model')
