import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc


class OmniglotTrain(Dataset):

    def __init__(self, data_path, size):
        super(OmniglotTrain, self).__init__()
        self.size = size
        alphabetas = [os.path.join(data_path, p) for p in os.listdir(data_path)]
        characters = []
        self.samples = []
        for alpha in alphabetas:
            characters.extend([os.path.join(alpha, p) for p in os.listdir(alpha)])
        for i in range(size):
            # produce data pairs from same characters
            if i%2 == 1:
                char = npc(characters, 1)[0]
                pair = npc(os.listdir(char), 2)
                self.samples.append((pair, 1.0))
            # produce data pairs from different characters
            else:
                chars = npc(characters, 2)
                pair = [os.listdir(char) for char in chars]
                self.samples.append((pair, 0.0))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.samples[index]


class OmniglotTest(Dataset):

    def __init__(self, data_path, way=20):
        super(OmniglotTest, self).__init__()
        self.way = way
        self.samples = []
        tmp = []
        path = [os.path.join(data_path, p) for p in os.listdir(data_path)]
        alphabetas = npc(path, 10)
        for time in range(2):
            for i, alpha in enumerate(alphabetas):
                cnt = 0
                for j, char in enumerate([os.path.join(alpha, p) for p in os.listdir(alpha)]):
                   cnt += 1
                   # print(os.path.join(char, p), 2)
                   tmp.append(npc([os.path.join(char, p) for p in os.listdir(char)], 2))
                   if cnt == self.way:
                       break

        for i in range(20):
            for j in range(self.way):
                for k in range(self.way):
                    idx1, idx2 = i * self.way + j, i * self.way + k
                    label = 1.0 if idx1 == idx2 else 0.0
                    self.samples.append(((tmp[idx1][0], tmp[idx2][1]), label))


    def __len__(self):
        return 20 * self.way

    def __getitem__(self, index):
        return self.samples[index]


# test
if __name__=='__main__':
    # omniglotTrain = OmniglotTrain('./images_background', 30000*8)
    # print(omniglotTrain)
    test = OmniglotTest('./images_evaluation', 20)
