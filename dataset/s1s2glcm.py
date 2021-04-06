import numpy as np
from PIL import Image
import os
import os.path

import torchvision
import torch
import torch.utils.data as data # only for checking dataloader, delete after
import torchvision.transforms as transforms

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

#not prepared for segmentation
'''
def get_s1s2glcm(root, transform_train=None, transform_val=None):

    def npy_loader(path):
        #sample = torch.from_numpy(np.load(path))
        sample = np.load(path)
        return sample

    trainvaldir = os.path.join(root, 'images/s1s2glcm/by-image/train+val')
    testdir = os.path.join(root, 'images/s1s2glcm/by-image/test')

    trainval_dataset = torchvision.datasets.DatasetFolder(root=trainvaldir,loader=npy_loader,transform=None,extensions=('.npy'))
    test_dataset = torchvision.datasets.DatasetFolder(root=testdir,loader=npy_loader,transform=None,extensions=('.npy'))

    print (f"#trainval: {len(trainval_dataset)} #test: {len(test_dataset)}")
    return trainval_dataset, test_dataset
'''

def get_s1s2glcm(root, transform_train=None, transform_val=None):

    def npy_loader(path):
        #sample = torch.from_numpy(np.load(path))
        sample = np.load(path)
        return sample

    datadir = os.path.join(root, 'images/s1s2glcm/by-image/train+val')
    maskdir = os.path.join(root, 'images/s1s2glcm/by-image/test')

    #split datadir into train/test
    train_idxs, test_idxs = train_val_split(base_dataset.targets)


    trainval_dataset = torchvision.datasets.DatasetFolder(root=trainvaldir,loader=npy_loader,transform=None,extensions=('.npy'))
    test_dataset = torchvision.datasets.DatasetFolder(root=testdir,loader=npy_loader,transform=None,extensions=('.npy'))

    print (f"#trainval: {len(trainval_dataset)} #test: {len(test_dataset)}")
    return trainval_dataset, test_dataset

class S1S2GLCM_labeled(torchvision.datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            indexs=None
    ) -> None:
        super(S1S2GLCM_labeled, self).__init__(root, loader=loader, extensions=extensions, transform=transform,
                                            target_transform=target_transform, is_valid_file=is_valid_file)
        if indexs is not None:
            new_samples = [self.samples[i] for i in indexs]
            self.samples=new_samples


            #self.targets = np.array(self.targets)[indexs]
            new_targets = [self.targets[i] for i in indexs]
            self.targets = new_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target