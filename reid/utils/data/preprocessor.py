from __future__ import absolute_import
import os.path as osp

from PIL import Image

import torch
class Preprocessor(object):
    def __init__(self, dataset, name=None,root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.name=name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')


        ####fg###
        # fgName=fname.split('.')[-2]+'.bmp'
        # fgpt='/home/fxyang/jjweng/CoCoPerdesGesture/mask/'+self.name+'/image/ImgFG'
        # fgPath=osp.join(fgpt,fgName)
        # img_fg=Image.open(fgPath)





        # bgName = fname.split('.')[-2] + '.bmp'
        # bgpt = '/home/fxyang/jjweng/CoCoPerdesGesture/mask/'+self.name+'/image/ImgBG'
        # bgPath = osp.join(bgpt, bgName)
        # img_bg = Image.open(bgPath)



        if self.transform is not None:
            img = self.transform(img)

            # img_fg=self.transform(img_fg)
            # img_bg=self.transform(img_bg)
            # img_fg=torch.zeros_like(img)
            # img_bg=torch.zeros_like(img)
        return img, fname, pid, camid
