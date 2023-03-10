import torch
from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
import cv2
import numpy
from pathlib import Path
from tqdm import tqdm
import functools


class PictureDataset(Dataset):
    def __init__(self):
        super(PictureDataset, self).__init__()
        self.frames = Path('frames')
        self.pos_dir = self.frames / '1_in_gaming'
        self.neg_dir = self.frames / '0_out_gaming'
        self.pos_images_paths = list(self.pos_dir.iterdir())
        self.neg_images_paths = list(self.neg_dir.iterdir())

    @functools.cached_property
    def pos_num(self):
        return len(self.pos_images_paths)

    @functools.cached_property
    def neg_num(self):
        return len(self.neg_images_paths)

    @functools.cached_property
    def __len__(self):
        return self.neg_num + self.pos_num

    def __getitem__(self, index):
        if index < self.neg_num:
            image_file = self.neg_images_paths[index]
            label = 0
        else:
            image_file = self.pos_images_paths[index - self.neg_num]
            label = 1

        image = cv2.imread(str(image_file))
        # default is (H, W, C), convert to (C, H, W)
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float)
        return image, label


class PictureBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=True):
        super(PictureBatchSampler, self).__init__(data_source)
        self.data_source = data_source  # data_source is an instance of PictureDataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        neg_sampler = iter(RandomSampler(range(self.data_source.neg_num)))
        pos_sampler = iter(RandomSampler(range(self.data_source.pos_num)))

        self.index_list = []
        neg_cnt, pos_cnt = 0, 0
        pos_neg_ratio = self.data_source.pos_num / self.data_source.neg_num
        batch = []
        while pos_cnt + neg_cnt < len(self.data_source):
            # get pos and neg samples in a batch according to total neg and pos samples in dataset
            if pos_cnt < pos_neg_ratio * neg_cnt and pos_cnt < self.data_source.pos_num:
                batch.append(next(pos_sampler) + self.data_source.neg_num)
                pos_cnt += 1
            else:
                batch.append(next(neg_sampler))
                neg_cnt += 1
            if len(batch) == self.batch_size:
                # print(list(map(lambda x: 'pos' if x > self.data_source.neg_num else 'neg', batch)))
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.data_source) // self.batch_size + int(not self.drop_last)


class GooseStopper(nn.Module):
    """
    use caught images to determine whether to stop or restart GooseReplay.
    """
    def __init__(self):
        super(GooseStopper, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=10, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=10, stride=2, padding=1, dilation=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=10, stride=2, padding=1, dilation=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(384, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Flatten(0),
            nn.Sigmoid()
        )

    def forward(self, image: Tensor):
        result = self.classifier(image)
        return result


if __name__ == '__main__':
    epochs = 10
    ds = PictureDataset()
    controller = GooseStopper()
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(controller.parameters(), lr=1e-4)  # lr is best set to 1e-4 or less
    best_f1 = 0
    for epoch in range(epochs):
        all_pred, all_truth = [], []
        for train_x, train_y in tqdm(DataLoader(ds, batch_sampler=PictureBatchSampler(ds, batch_size=32), num_workers=4)):
            pred = controller.forward(train_x)
            all_pred.extend(pred.tolist())
            all_truth.extend(train_y)
            loss = criterion(pred, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_pred, all_truth = torch.tensor(all_pred), torch.tensor(all_truth).to(torch.int32)
        all_pred = torch.where(all_pred > 0.5, 1, 0)
        print(f'\naccuracy: {all_pred.eq(all_truth).sum() / all_pred.size(0):.2%}')
        recall = all_pred.masked_select(all_truth.to(torch.bool)).sum() / all_truth.sum()
        print(f'recall: {recall:.2%}')
        precision = all_truth.masked_select(all_pred.to(torch.bool)).sum() / all_pred.sum()
        print(f'precision: {precision:.2%}')
        f1 = 2 * recall * precision / (precision + recall)
        print(f'F1-Score: {f1:.2%}')

        if f1 > best_f1:  # save best f1 model
            best_f1 = f1
            print(f'saving ...')
            torch.save(controller, f'goose_stopper_{best_f1:.2}.th')
