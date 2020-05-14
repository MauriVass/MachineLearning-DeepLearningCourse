from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset(directory, file, class_to_idx):
    instances_train = []
    instances_val = []
    # Expand the environment variables
    # with their corresponding
    # value in the given paths
    #directory = os.path.expanduser(directory)

    root = directory.split('/')[0]
    indexes = open(f'{root}/{ file }.txt','r').readlines()
    count = 0
    for i in indexes:
        i = i.replace('\n','')
        path = os.path.join(directory, i)
        class_name = i.split('/')[0]
        if(class_name != 'BACKGROUND_Google'):
            class_index = class_to_idx[class_name]
            item = path, class_index
            if(count%2==0):
                #Validation
                instances_val.append(item)
            else:
                #Train
                instances_train.append(item)
            count += 1
    return instances_train , instances_val


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class)
        '''

        classes, class_to_idx = self._find_classes(self.root)
        samples_train, samples_val = make_dataset(self.root, split, class_to_idx)#
        if len(samples_train) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples_train = samples_train
        self.samples_val = samples_val
        self.samples = []#samples_train + samples_val

        #Get the class from each element
        self.targets_train = [s[1] for s in samples_train]
        self.targets_val = [s[1] for s in samples_val]
        self.targets = []#[s[1] for s in self.samples]
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """

        classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name != 'BACKGROUND_Google']
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        #Image Tensor and class label
        return sample, target

    def __getitemByPath__(self, path):
        '''
        __getitem__ should access an element through its path
        Args:
            path (string): Path
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        target = -1
        sample = pil_loader(path)
        for s in self.samples:
          if s[0] == path:
            target = s[1]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        #Image Tensor and class label
        return sample, target

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.samples) # Provide a way to get the length (number of elements) of the dataset
        return length


    def SetVal(self):
        self.samples = self.samples_val
        self.targets = self.targets_val
    def SetTrain(self):
        self.samples = self.samples_train
        self.targets = self.targets_train
    def SetTest(self):
        self.samples = self.samples_train + self.samples_val
        self.targets = self.targets_train + self.targets_val

    def getClass(self):
        return self.classes
    def getClass_to_idx(self):
        return self.class_to_idx
    def getTarget(self):
        return self.targets
    def getSample(self):
        return self.samples
    def getRoot(self):
        return self.root
