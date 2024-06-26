import pydicom
import os
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import random 
import zarr
import lightning as L
import torch.nn as nn
import torch
from models.navit import NaViT
from torch.nn import BCELoss
import pandas as pd


class TrainDataset(IterableDataset):
    def __init__(self, train_series, batch_size=10):
        self.num_batches = 1000
        self.batch_size = batch_size
        self.series_metadata = pd.read_json('/cbica/home/gangarav/projects/rsna_lumbar/axial_t2_data.json')

        self.train_series = train_series

    def __iter__(self):
        for _ in range(self.num_batches):    
            x, y, z = self._choose_three_numbers_sum_to_18()
            batch = []

            for _ in range(self.batch_size):

                zarr_ref = None
                enough_frames = False
                while not enough_frames:
                    series_path = random.choice(self.train_series)
                    series = self._get_series_from_path(series_path)
                    enough_frames = self._enough_dims_in_series(series, (x, y, z))

                    if not enough_frames:
                        continue

                    zarr_ref = self._get_zarr_reference(series_path)

                    slice_indices_1 = self._get_frame_indices_for_zarr_reference_and_slice_shape(series, (x, y, z))
                    slice_indices_2 = self._get_frame_indices_for_zarr_reference_and_slice_shape(series, (x, y, z))

                    slices_1 = self._get_frames_from_zarr_reference(zarr_ref, slice_indices_1)
                    slices_2 = self._get_frames_from_zarr_reference(zarr_ref, slice_indices_2)

                    midpoint_1 = np.array([np.mean(i_pairs) for i_pairs in slice_indices_1])
                    midpoint_2 = np.array([np.mean(i_pairs) for i_pairs in slice_indices_2])

                    vector = midpoint_1 - midpoint_2

                    # check if absolute value of vector[0] is less than 2
                    if abs(vector[0]) < 5:
                        enough_frames = False

                vector = [0.0 if value < 0 else 1.0 for value in vector]


                batch.append((slices_1, slices_2, [vector[0]]))

            yield tuple(np.stack(t) for t in zip(*batch))

    def _get_series_from_path(self, path):
        return int(path.split('/')[-1].split(".")[0])

    def _choose_three_numbers_sum_to_18(self, min=3):
        x = random.randint(0, 5)
        y = random.randint(9 - x, 9)
        z = 18 - x - y
        return 1, 256, 256
        return 2**x, 2**y, 2**z

    def _get_list_of_valid_studies(self):
        # Implement logic to get the list of valid studies
        pass

    def _get_zarr_reference(self, series_path):
        return zarr.open(series_path, mode='r')

    def _get_frames_from_zarr_reference(self, zarr, slice_list=None):
        if slice_list is not None:
            slices = tuple(slice(start, end) for start, end in slice_list)
            z = zarr[slices]
        else:
            z = zarr[:]
        z = z - np.mean(z)
        z = z/np.std(z)
        return np.expand_dims(z, axis=0)

    def _get_frame_indices_for_zarr_reference_and_slice_shape(self, series_id, slice_shape):
        slice_indices = []
        dims = self._get_dims_from_series(series_id)
        for axis, size in enumerate(dims):
            if size < slice_shape[axis]:
                return None
            else:
                index = random.randint(0, size - slice_shape[axis])
                slice_indices.append([index, index + slice_shape[axis]])
        return slice_indices

    def _get_dims_from_series(self, series_id):
        row = self.series_metadata.loc[self.series_metadata["series_id"] == series_id]
        return row.z_dim.values[0], row.x_dim.values[0], row.y_dim.values[0]

    def _enough_dims_in_series(self, series_id, slice_shape):
        dims = self._get_dims_from_series(series_id)
        for axis, size in enumerate(dims):
            if size < slice_shape[axis]:
                return False
        return True


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch = NaViT(
                image_size = (8, 512, 512),
                patch_size = (1, 16, 16),
                num_classes = 10,
                dim = 768,
                depth = 12,
                heads = 12,
                mlp_dim = 3072,
                dropout = 0.1,
                emb_dropout = 0.1,
                #token_dropout_prob = 0.1  # token dropout of 10% (keep 90% of tokens)
            )
        self.trunk = nn.Sequential(
            nn.Linear(768*2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

    # p1 = v(c[0], group_images = True, group_max_seq_len = 128)
    # p2 = v(c[0], group_images = True, group_max_seq_len = 128)
    # p = torch.cat((p1, p2), 1)
    # t = conjoined_layer(p)
    # o = sigmoid(t)


    def forward_once(self, x):
        output = self.branch(x, group_images = True, group_max_seq_len = 512)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.trunk(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output

class EndToEnd(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = SiameseNetwork()
        self.loss = BCELoss()

    def forward(self, inputs_1, inputs_2):
        return self.net(inputs_1, inputs_2)

    def training_step(self, batch, batch_idx):
        inputs_1, inputs_2, label = batch
        label = torch.where(label > 0, torch.tensor(1.0), torch.tensor(0.0))
        label = label.type(torch.FloatTensor)

        inputs_1 = inputs_1.to('cuda:0')
        inputs_2 = inputs_2.to('cuda:0')
        label = label.to('cuda:0')

        fo = self(inputs_1, inputs_2)

        #r_loss = self.recon_loss(autoenc_v1, og_inputs_1[:, 0:1, :, :])
        fo_loss = self.loss(fo, label)
        self.log("loss", fo_loss, prog_bar=True)
        total_loss =  fo_loss
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(list(self.net.parameters()), lr=1e-3)

class DataModule(L.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        zarr_dir = '/cbica/home/gangarav/rsna24_preprocessed/'
        self.train_studies = [f"{zarr_dir}{file}" for file in os.listdir(zarr_dir)]

    def train_dataloader(self):
        dataset = TrainDataset(self.train_studies, self.batch_size)
        dataloader = DataLoader(dataset, batch_size=None, num_workers=4)
        return dataloader

dm = DataModule(batch_size=10)
sn = EndToEnd()
trainer = L.Trainer(max_epochs=10000, accumulate_grad_batches=100)
# tuner = Tuner(trainer)
# tuner.scale_batch_size(sn, datamodule=dm, mode="binsearch")

trainer.fit(sn, datamodule=dm)