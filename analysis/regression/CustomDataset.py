import torch
import numpy as np
from RandomCrop import RandomCrop
from torchvision.transforms.functional import center_crop


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, data_aug):
        super(CustomDataset).__init__()
        self.dataset = dataset
        self.data_aug = data_aug
        self.strong_typhoon_list = []
        self.mult_proportion = {35.0: 0, 40.0: 0, 45.0: 0, 50.0: 0, 55.0: 0, 60.0: 1,
                                65.0: 1, 70.0: 1, 75.0: 1, 80.0: 1, 85.0: 2, 90.0: 2,
                                95.0: 4, 100.0: 6, 105.0: 14, 110.0: 17, 115.0: 39,
                                120.0: 150, 125.0: 200, 130.0: 200, 135.0: 0, 140.0: 200,}
        if self.data_aug:
            for idx in range(len(self.dataset)):
                _, labels_0 = self.dataset[idx]
                for _ in range(self.mult_proportion[labels_0]):
                    self.strong_typhoon_list.append(idx)

                if idx%1000 == 0:
                    print(idx)
            np.save("strong_typhoon_list3.npy", self.strong_typhoon_list)
            # self.strong_typhoon_list = np.load("strong_typhoon_list3.npy")

    def __getitem__(self, idx):
        if idx < len(self.dataset):
            image_0, labels_0 = self.dataset[idx]
        else:
            image_0, labels_0 = self.dataset[self.strong_typhoon_list[idx-len(self.dataset)]]

        image_tensor = torch.tensor(image_0)
        if self.data_aug:
            image_tensor = RandomCrop(224)(image_tensor)
        else:
            image_tensor = center_crop(image_tensor, (224, 224))

        return image_tensor, labels_0

    def __len__(self):
        if self.data_aug:
            return (len(self.dataset) + len(self.strong_typhoon_list))
        else:
            return len(self.dataset)
