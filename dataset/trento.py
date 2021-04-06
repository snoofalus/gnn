import numpy as np
from PIL import Image
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torchvision
import torch
import torch.utils.data as data # only for checking dataloader, delete after
import torchvision.transforms as transforms
import torch.utils.data.dataset as Dataset  # For custom data-sets

#DONTCOMMIT prepare trento

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

def get_trento(root):

	#DONTCOMMIT add transformations and mu, std of dataset for norm here
	norm = np.load(os.path.join(root,'workdir/trento/trento_norm.npy'))

	image_transform = transforms.Compose([
		ToTensor(),
		transforms.Normalize(norm[:, 0], norm[:, 1])])


	def npy_loader(path):

		#do once either here or in ToTensor
		#sample = torch.from_numpy(np.load(path))

		image = np.load(path)
		return image

	#root is start of datafolder e.g. "../data-local", already abspath
	data_dir = os.path.join(root, 'images/trento/by-image/data')
	mask_dir = os.path.join(root, 'images/trento/by-image/mask')

	#DONTCOMMIT alternative to doing inside Dataset
	#glob with wildcard* gets all files with extension
	#data_paths = glob.glob("D:\\Neda\\Pytorch\\U-net\\BMMCdata\\data\\*.jpg")
	#mask_paths = glob.glob("D:\\Neda\\Pytorch\\U-net\\BMMCdata\\masks\\*.jpg")

	trento_dataset = SegmentationDataset(root_image=data_dir, root_mask=mask_dir, loader=npy_loader, image_transform=image_transform, mask_transform=None)


	#print (f"#trainval: {len(trainval_dataset)} #test: {len(test_dataset)}")
	return trento_dataset#trainval test


###
#Custom segmentation dataset
###-------------------------------------------------------------------------

#DONTCOMMIT REMOVE THIS SECTION IF NO ERRORS
'''
def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
	"""Checks if a file is an allowed extension.

	Args:
	    filename (string): path to a file
	    extensions (tuple of strings): extensions to consider (lowercase)

	Returns:
	    bool: True if the filename ends with one of given extensions
	"""
	return filename.lower().endswith(extensions)



def is_image_file(filename: str) -> bool:
	"""Checks if a file is an allowed image extension.

	Args:
	    filename (string): path to a file

	Returns:
	    bool: True if the filename ends with a known image extension
	"""
	return has_file_allowed_extension(filename, IMG_EXTENSIONS)
'''

def make_dataset(image_directory: str, mask_directory: str,): #-> List[Tuple[str, int]]:
	image_paths = []
	mask_paths = []
	image_directory = os.path.expanduser(image_directory)
	mask_directory = os.path.expanduser(mask_directory)

	'''DONTCOMMIT REMOVE THIS IF NO ERRORS
	both_none = extensions is None and is_valid_file is None
	both_something = extensions is not None and is_valid_file is not None
	if both_none or both_something:
		raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
	if extensions is not None:
		def is_valid_file(x: str) -> bool:
			return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

	is_valid_file = cast(Callable[[str], bool], is_valid_file)
	for target_class in sorted(class_to_idx.keys()):
		class_index = class_to_idx[target_class]
		target_dir = os.path.join(directory, target_class)
		if not os.path.isdir(target_dir):
			continue
	'''

	for root, _, fnames in sorted(os.walk(image_directory, followlinks=True)):
		for fname in sorted(fnames):
			path = os.path.join(root, fname)
			#if is_valid_file(path):
			image_paths.append(path)

	for root, _, fnames in sorted(os.walk(mask_directory, followlinks=True)):
		for fname in sorted(fnames):
			path = os.path.join(root, fname)
			#if is_valid_file(path):
			mask_paths.append(path)


	return image_paths, mask_paths


class SegmentationDataset(data.Dataset):

	'''
	Your custom dataset should inherit Dataset and override the following methods:

		__len__ so that len(dataset) returns the size of the dataset.
		__getitem__ to support the indexing such that dataset[i] can be used to get i'th sample
	'''

	#store just image paths in init

	def __init__(self, root_image: str, root_mask: str, loader: Callable[[str], Any], image_transform: Optional[Callable] = None, mask_transform: Optional[Callable] = None, is_valid_file: Optional[Callable[[str], bool]] = None) -> None:
		super(SegmentationDataset, self).__init__()

		image_paths, mask_paths = make_dataset(root_image, root_mask)

		self.loader = loader

		self.image_paths = image_paths
		self.mask_paths = mask_paths

		self.image_transform = image_transform
		self.mask_transform = mask_transform

	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, mask)
		"""
		image_path = self.image_paths[index]
		mask_path = self.mask_paths[index]

		image = self.loader(image_path)
		mask = self.loader(mask_path)

		if self.image_transform is not None:
			image = self.image_transform(image)
		if self.mask_transform is not None:
			mask = self.mask_transform(mask)

		return image, mask

	def __len__(self) -> int:
		return len(self.image_paths)
