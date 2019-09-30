import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms

class LinearAct(nn.Module):
    def __init__(self, fan_in, fan_out, act='linear', bias=True):
        super().__init__()
        self.fc = nn.Linear(fan_in, fan_out, bias)

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'linear':
            self.act = None
        else:
            assert False

        nn.init.kaiming_uniform_(self.fc.weight, a=0, mode='fan_in', nonlinearity=act)

    def forward(self, x):
        x = self.fc(x)
        if self.act is not None:
            x = self.act(x)
        return x

class BufferLoader:
    def __init__(self, load_func=None, proc_func=None, save_file=None,
                 save_per_new=1000):
        self.save_file = save_file
        self.load_func = load_func
        self.proc_func = proc_func
        self.new_ids = 0
        self.save_per_new = save_per_new

        if save_file != None:
            self.buffer = pickle.load(open(save_file, 'rb'))
        else:
            self.buffer = {}

        if self.load_func is None:
            self.load_func = open


    def load(self, path):
        """
        Load a file from path
        :param path:
        :return:  if proc_func, return proc_func(load_func(path))
        """

        # If the path is buffered
        if path in self.buffer:
            return self.buffer[path]

        content = self.load_func(path)

        if self.proc_func is not None:
            content = self.proc_func(content)

        self.buffer[path] = content

        if self.save_file is not None:
            self.new_ids += 1
            if self.new_ids >= self.save_per_new:
                pickle.dump(self.buffer, open(self.save_file, 'wb'))
                self.new_ids = 0

        return content

    def dump_buffer(self):
        pickle.dump(self.buffer, open(self.save_file, 'wb'))


def cut_inst_with_leng(inst, leng):
    """
    inst = inst[:, max(leng)]
    """
    return inst[:, :max(leng)]

from PIL import Image
import numpy as np
def pil_saver(img, path):
    # print(img, path)
    with open(path, 'wb') as f:
        Image.fromarray(img, 'RGB').save(f)
denormalize = transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1. / 0.229, 1. / 0.224, 1. / 0.225])
denormalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1. / 0.229, 1. / 0.224, 1. / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                         std=[1., 1., 1.]),
])
def show_case(src, trg, inst, tok=None, save_to='tmp/tmp'):
    plt.imsave("%s_src.jpg" % save_to, denormalize(src).permute(1, 2, 0))
    plt.imsave("%s_trg.jpg" % save_to, denormalize(trg).permute(1, 2, 0))
    # pil_saver(denormalize(src).numpy(), "%s_src.jpg" % save_to)
    # pil_saver(denormalize(trg).numpy(), "%s_trg.jpg" % save_to)
    with open("%s.txt" % save_to, 'w') as f:
        if tok is not None:
            inst = tok.decode(tok.shrink(inst.numpy()))
        f.write("%s\n" % inst)

