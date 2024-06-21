import marimo

__generated_with = "0.6.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pydicom
    import os
    import numpy as np

    root_dir = '/cbica/home/gangarav/projects/rsna_lumbar/raw'
    study_paths = []
    series_paths = []

    for study_name in os.listdir(root_dir):
        study_path = os.path.join(root_dir, study_name)
        study_paths.append(study_path)

        for series_name in os.listdir(study_path):
            series_path = os.path.join(study_path, series_name)
            series_paths.append(series_path)

    print("Study Paths:")
    print(len(study_paths), study_paths)
    print("\nSeries Paths:")
    print(len(series_paths), series_paths)


    def load_dicom(path):
        files = [(pydicom.dcmread(f"{path}/{f}").InstanceNumber, f) for f in os.listdir(f"{path}")]
        files = sorted(files,  key=lambda x: x[0])
        frames = []
        for _, f in files:
            ds = pydicom.dcmread(f"{path}/{f}")
            frames.append(np.expand_dims(ds.pixel_array, (0)))
        s = np.vstack(frames)
        return s.astype(np.float32), np.round(ds.ImageOrientationPatient, 0)

    y = [load_dicom(x) for x in series_paths]
    return (
        load_dicom,
        mo,
        np,
        os,
        pydicom,
        root_dir,
        series_name,
        series_path,
        series_paths,
        study_name,
        study_path,
        study_paths,
        y,
    )


@app.cell
def __(y):
    [z[0].shape[1] for z in y]
    return


@app.cell
def __(np):
    from torch.utils.data import IterableDataset, DataLoader
    import random 
    import zarr

    from monai.transforms import (
        Resize
    )

    class TrainDataset(IterableDataset):
        def __init__(self, train_studies):
            self.num_batches = 100
            self.batch_size = 20

            self.studies = train_studies

        def __iter__(self):
            for _ in range(self.num_batches):    
                x, y, z = self._choose_three_numbers_sum_to_18()
                batch = []

                for _ in range(self.batch_size):

                    zarr_ref = None
                    enough_frames = False
                    while not enough_frames:
                        study = random.choice(self.studies)
                        zarr_ref = self._get_zarr_reference(study)
                        enough_frames = self._enough_frames_in_zarr_reference(zarr_ref, (x, y, z))

                    slice_indices_1 = self._get_frame_indices_for_zarr_reference_and_slice_shape(zarr_ref, (x, y, z))
                    slice_indices_2 = self._get_frame_indices_for_zarr_reference_and_slice_shape(zarr_ref, (x, y, z))

                    slices_1 = self._get_frames_from_zarr_reference(zarr_ref, slice_indices_1)
                    slices_2 = self._get_frames_from_zarr_reference(zarr_ref, slice_indices_2)

                    midpoint_1 = np.array([np.mean(i_pairs) for i_pairs in slice_indices_1])
                    midpoint_2 = np.array([np.mean(i_pairs) for i_pairs in slice_indices_2])

                    vector = midpoint_1 - midpoint_2

                    # vector = [0.0 if vector[0] < 0 else 1.0]

                    batch.append((slices_1, slices_2, [vector[0]]))

                yield tuple(np.stack(t) for t in zip(*batch))

        def _choose_three_numbers_sum_to_18(self, min=3):
            x = random.randint(0, 5)
            y = random.randint(9 - x, 9)
            z = 18 - x - y
            return 1, 256, 256
            return 2**x, 2**y, 2**z

        def _get_list_of_valid_studies(self):
            # Implement logic to get the list of valid studies
            pass

        def _get_zarr_reference(self, study):
            array = None
            with zarr.DirectoryStore(f'{study}') as store:
                array = zarr.open(store, mode='r')
            return array

        def _get_frames_from_zarr_reference(self, zarr, slice_list=None):
            if slice_list is not None:
                slices = tuple(slice(start, end) for start, end in slice_list)
                zarr = zarr[slices]

            #T = Resize((256, 256), mode="bilinear")
            return np.expand_dims(zarr[:], axis=0)

        def _get_frame_indices_for_zarr_reference_and_slice_shape(self, zarr, slice_shape):
            slice_indices = []
            for axis, size in enumerate(zarr.shape):
                if size < slice_shape[axis]:
                    return None
                else:
                    index = random.randint(0, size - slice_shape[axis])
                    slice_indices.append([index, index + slice_shape[axis]])
            return slice_indices

        def _enough_frames_in_zarr_reference(self, zarr, slice_shape):
            for axis, size in enumerate(zarr.shape):
                if size < slice_shape[axis]:
                    return False
            return True
    return DataLoader, IterableDataset, Resize, TrainDataset, random, zarr


@app.cell
def __(DataLoader, TrainDataset):
    a = TrainDataset(["/cbica/home/gangarav/rsna24_preprocessed/1012284084.zarr/"])
    dataloader = DataLoader(a, batch_size=None, num_workers=1)
    b = iter(dataloader)
    c = next(b)
    return a, b, c, dataloader


@app.cell
def __():
    import lightning as L
    import torch.nn as nn
    import torch
    from models.navit import NaViT
    from torch.nn import BCELoss

    class SiameseNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            self.branch = NaViT(
                    image_size = (8, 512, 512),
                    patch_size = (1, 16, 16),
                    num_classes = 10,
                    dim = 1024,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 2048,
                    dropout = 0.1,
                    emb_dropout = 0.1,
                    #token_dropout_prob = 0.1  # token dropout of 10% (keep 90% of tokens)
                )
            self.trunk = nn.Sequential(
                nn.Linear(20, 256),
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
            output = self.branch(x, group_images = True, group_max_seq_len = 256)
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
    return BCELoss, EndToEnd, L, NaViT, SiameseNetwork, nn, torch


@app.cell
def __(DataLoader, L, TrainDataset):
    class DataModule(L.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.train_studies = ["/cbica/home/gangarav/rsna24_preprocessed/1012284084.zarr/"]

        def train_dataloader(self):
            dataset = TrainDataset(self.train_studies)
            dataloader = DataLoader(dataset, batch_size=None, num_workers=1)
            return dataloader
    return DataModule,


@app.cell
def __(DataModule, EndToEnd, L):
    dm = DataModule()
    sn = EndToEnd()
    trainer = L.Trainer(max_epochs=1000)

    # p1 = v(c[0], group_images = True, group_max_seq_len = 128)
    # p2 = v(c[0], group_images = True, group_max_seq_len = 128)
    # p = torch.cat((p1, p2), 1)
    # t = conjoined_layer(p)
    # o = sigmoid(t)
    return dm, sn, trainer


@app.cell
def __(dm, sn, trainer):
    trainer.fit(sn, datamodule=dm)
    return


@app.cell
def __(c, mo):
    i = mo.ui.slider(0, c[0].shape[2] - 1, orientation='vertical')
    j = mo.ui.slider(0, c[0].shape[0] - 1)
    return i, j


@app.cell
def __(c):
    c[0].shape
    return


@app.cell
def __(c, i, j, mo):
    mo.vstack([
        mo.md("##Batch Visualizer"),
        mo.hstack([
            j,
            mo.md(f"###batch item: {j.value}")
        ], justify="start"),
        mo.hstack([
            i,
            mo.image(src=c[0][j.value][0,i.value,:,:]),
            mo.image(src=c[1][j.value][0,i.value,:,:]),
            mo.vstack([
                mo.plain_text(f"the left image is {'superior' if c[2][j.value][0] < 0 else 'inferior'} to the right"),
                mo.plain_text(f"the left image is {'anterior' if c[2][j.value][1] < 0 else 'posterior'} to the right"),
                mo.plain_text(f"the left image is radiology {'right' if c[2][j.value][2] < 0 else 'left'} of the right"),
            ])
        ], justify="start"),
    ])
    return


@app.cell
def __(NaViT, batch_norm_to_group_norm, nn, torch):
    class SiameseTwinSegmentationNetwork(nn.Module):

        def __init__(self):
            super(SiameseTwinSegmentationNetwork, self).__init__()
            # self.FC_DROPOUT = 0.1 #options["FC_DROPOUT"]
            self.ACTIVATION = "RELU" #options["ACTIVATION"]

            self.model = NaViT(
                image_size = (8, 512, 512),
                patch_size = (1, 16, 16),
                num_classes = 1000,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1,
                #token_dropout_prob = 0.1  # token dropout of 10% (keep 90% of tokens)
            )

            self.model.encoder = batch_norm_to_group_norm(self.model.encoder)

            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

            FC_IN = 2048
            # if options["MODEL"] in ["resnet18", "resnet34"]:
            #     FC_IN = 512

            self.conjoined_layer = nn.Sequential(
                nn.Dropout(p=self.FC_DROPOUT, inplace=True),
                nn.Linear(FC_IN * 2, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
            )

            self.sigmoid = nn.Sigmoid()

        def forward_once(self, x):
            output = self.model.encoder(x)
            output = output[-1]
            output = self.classification_head(output)
            return output

        def trunk(self, input1, input2):
            output = torch.cat((input1, input2), 1)

            # pass the concatenation to the linear layers
            output = self.conjoined_layer(output)

            # pass the out of the linear layers to sigmoid layer
            output = self.sigmoid(output)

            return output

        def forward(self, input1, input2):
            # get two images' features
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            # concatenate both images' features
            output = torch.cat((output1, output2), 1)

            # pass the concatenation to the linear layers
            output = self.conjoined_layer(output)

            # pass the out of the linear layers to sigmoid layer
            output = self.sigmoid(output)

            return output
    return SiameseTwinSegmentationNetwork,


@app.cell
def __(NaViT):
    v = NaViT(
        image_size = (8, 512, 512),
        patch_size = (1, 16, 16),
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        #token_dropout_prob = 0.1  # token dropout of 10% (keep 90% of tokens)
    )

    # 5 images of different resolutions - List[List[Tensor]]

    # for now, you'll have to correctly place images in same batch element as to not exceed maximum allowed sequence length for self-attention w/ masking
    import torch
    images = [
        torch.randn(1, 3, 128, 64),
        torch.randn(1, 1, 64, 512)
    ]

    preds = v(
        images,
        group_images = True,
        group_max_seq_len = 14
    )

    print(preds.shape)
    return images, preds, torch, v


@app.cell
def __():
    from torch import optim, nn, utils, Tensor
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    import lightning as L
    import torch
    return L, MNIST, Tensor, ToTensor, nn, optim, torch, utils


@app.cell
def __():
    return


@app.cell
def __():
    # import random
    # import zarr


    # from monai.transforms import (
    #     Resize
    # )

    # class CustomIterableDataset(IterableDataset):
    #     def __init__(self):
    #         self.num_batches = 1000
    #         self.batch_size = 10

    #         self.studies = os.listdir('/cbica/home/gangarav/rsna24_preprocessed/')

    #     def __iter__(self):
    #         for _ in range(self.num_batches):    
    #             x, y, z = self._choose_three_numbers_sum_to_18()
    #             batch = []

    #             for _ in range(self.batch_size):
    #                 study = random.choice(self.studies)
    #                 zarr_ref = None
    #                 enough_frames = False
    #                 while not enough_frames:
    #                     zarr_ref = self._get_zarr_reference(study)
    #                     enough_frames = self._enough_frames_in_zarr_reference(zarr_ref, (x, y, z))

    #                 slice_indices_1 = self._get_frame_indices_for_zarr_reference_and_slice_shape(zarr_ref, (x, y, z))
    #                 slice_indices_2 = self._get_frame_indices_for_zarr_reference_and_slice_shape(zarr_ref, (x, y, z))

    #                 print(slice_indices_1)
    #                 print(slice_indices_2)

    #                 slices_1 = self._get_frames_from_zarr_reference(zarr_ref, slice_indices_1)
    #                 slices_2 = self._get_frames_from_zarr_reference(zarr_ref, slice_indices_2)

    #                 midpoint_1 = np.array([np.mean(i_pairs) for i_pairs in slice_indices_1])
    #                 midpoint_2 = np.array([np.mean(i_pairs) for i_pairs in slice_indices_2])

    #                 vector = midpoint_1 - midpoint_2

    #                 vector = [0.0 if vector[0] < 0 else 1.0]

    #                 batch.append((slices_1, slices_2, vector))

    #             yield tuple(np.stack(t) for t in zip(*batch))

    #     def _choose_three_numbers_sum_to_18(self, min=3):
    #         x = random.randint(0, 5)
    #         y = random.randint(9 - x, 9)
    #         z = 18 - x - y
    #         return 1, 200, 200
    #         return 2**x, 2**y, 2**z

    #     def _get_list_of_valid_studies(self):
    #         # Implement logic to get the list of valid studies
    #         pass

    #     def _get_zarr_reference(self, study):
    #         array = None
    #         with zarr.DirectoryStore(f'/cbica/home/gangarav/rsna24_preprocessed/{study}') as store:
    #             array = zarr.open(store, mode='r')
    #         return array

    #     def _get_frames_from_zarr_reference(self, zarr, slice_list=None):
    #         if slice_list is not None:
    #             slices = tuple(slice(start, end) for start, end in slice_list)
    #             zarr = zarr[slices]

    #         T = Resize((128, 128), mode="bilinear")
    #         return T(zarr[:])

    #     def _get_frame_indices_for_zarr_reference_and_slice_shape(self, zarr, slice_shape):
    #         slice_indices = []
    #         for axis, size in enumerate(zarr.shape):
    #             if size < slice_shape[axis]:
    #                 return None
    #             else:
    #                 index = random.randint(0, size - slice_shape[axis])
    #                 slice_indices.append([index, index + slice_shape[axis]])
    #         return slice_indices

    #     def _enough_frames_in_zarr_reference(self, zarr, slice_shape):
    #         for axis, size in enumerate(zarr.shape):
    #             if size < slice_shape[axis]:
    #                 return False
    #         return True
    return


@app.cell
def __(CustomIterableDataset, DataLoader, L):
    class CustomDataModule(L.LightningDataModule):
        def __init__(self):
            super().__init__()

        def train_dataloader(self):
            dataset = CustomIterableDataset()
            dataloader = DataLoader(dataset, batch_size=None, num_workers=4)
            return dataloader
    return CustomDataModule,


@app.cell
def __(CustomIterableDataset):
    dataset = CustomIterableDataset()
    iterator = iter(dataset)
    first_sample = next(iterator)
    return dataset, first_sample, iterator


@app.cell
def __(first_sample):
    import matplotlib.pyplot as plt
    first_sample[0].shape
    return plt,


@app.cell
def __(first_sample, mo):
    x = mo.ui.slider(0, first_sample[0].shape[0] - 1)
    x
    mo.hstack(
        [

        ]
    )
    return x,


@app.cell
def __(first_sample, mo, plt, x):
    # plt.imshow(image_data, cmap='gray', aspect='auto', extent=[0,10,0,10])
    # plt.axis('off')  # Turn off the axis
    # plt.tight_layout()  # Adjust the layout to minimize whitespace
    # plt.show()

    mo.hstack(
        [
            mo.vstack([
                plt.imshow(first_sample[0][x.value][0]),
            ]),
            mo.vstack([
                plt.imshow(first_sample[1][x.value][0]),
            ]),
        ]
    )

    # image_data1 = first_sample[0][x.value][0]
    # image_data2 = first_sample[1][x.value][0]

    # # Set up the figure and subplots
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # # Plot the first image
    # ax1 = axes[0]
    # ax1.imshow(image_data1, cmap='gray', aspect='auto')
    # ax1.axis('off')  # Turn off the axis

    # # Plot the second image
    # ax2 = axes[1]
    # ax2.imshow(image_data2, cmap='gray', aspect='auto')
    # ax2.axis('off')  # Turn off the axis

    # # Adjust layout
    # plt.tight_layout()
    return


@app.cell
def __():
    #from vit_pytorch.na_vit import NaViT, group_images_by_max_seq_len
    return


@app.cell
def __(Tensor, nn, torch):
    from functools import partial
    from typing import List, Union

    #import torch
    import torch.nn.functional as F
    #from torch import nn, Tensor
    from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange

    # helpers

    def exists(val):
        return val is not None

    def default(val, d):
        return val if exists(val) else d

    def always(val):
        return lambda *args: val

    def pair(t):
        return t if isinstance(t, tuple) else (t, t)

    def divisible_by(numer, denom):
        return (numer % denom) == 0

    # auto grouping images

    def group_images_by_max_seq_len(
        images: List[Tensor],
        patch_size: int,
        calc_token_dropout = None,
        max_seq_len = 2048

    ) -> List[List[Tensor]]:

        calc_token_dropout = default(calc_token_dropout, always(0.))

        groups = []
        group = []
        seq_len = 0

        if isinstance(calc_token_dropout, (float, int)):
            calc_token_dropout = always(calc_token_dropout)

        for image in images:
            assert isinstance(image, Tensor)

            # VG: 3d mod
            image_dims = image.shape[-2:]
            # VG: 3d mod
            ph, pw = map(lambda t: t // patch_size, image_dims)

            # VG: 3d mod
            image_seq_len = (ph * pw)
            image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

            assert image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'

            if (seq_len + image_seq_len) > max_seq_len:
                groups.append(group)
                group = []
                seq_len = 0

            group.append(image)
            seq_len += image_seq_len

        if len(group) > 0:
            groups.append(group)

        return groups

    # normalization
    # they use layernorm without bias, something that pytorch does not offer

    class LayerNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(dim))
            self.register_buffer('beta', torch.zeros(dim))

        def forward(self, x):
            return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

    # they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper

    class RMSNorm(nn.Module):
        def __init__(self, heads, dim):
            super().__init__()
            self.scale = dim ** 0.5
            self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

        def forward(self, x):
            normed = F.normalize(x, dim = -1)
            return normed * self.scale * self.gamma

    # feedforward

    def FeedForward(dim, hidden_dim, dropout = 0.):
        return nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    class Attention(nn.Module):
        def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
            super().__init__()
            inner_dim = dim_head *  heads
            self.heads = heads
            self.norm = LayerNorm(dim)

            self.q_norm = RMSNorm(heads, dim_head)
            self.k_norm = RMSNorm(heads, dim_head)

            self.attend = nn.Softmax(dim = -1)
            self.dropout = nn.Dropout(dropout)

            self.to_q = nn.Linear(dim, inner_dim, bias = False)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim, bias = False),
                nn.Dropout(dropout)
            )

        def forward(
            self,
            x,
            context = None,
            mask = None,
            attn_mask = None
        ):
            x = self.norm(x)
            kv_input = default(context, x)

            qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

            q = self.q_norm(q)
            k = self.k_norm(k)

            dots = torch.matmul(q, k.transpose(-1, -2))

            if exists(mask):
                mask = rearrange(mask, 'b j -> b 1 1 j')
                dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

            if exists(attn_mask):
                dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)

    class Transformer(nn.Module):
        def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
            super().__init__()
            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
                ]))

            self.norm = LayerNorm(dim)

        def forward(
            self,
            x,
            mask = None,
            attn_mask = None
        ):
            for attn, ff in self.layers:
                x = attn(x, mask = mask, attn_mask = attn_mask) + x
                x = ff(x) + x

            return self.norm(x)

    class NaViT(nn.Module):
        def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., token_dropout_prob = None):
            super().__init__()
            image_height, image_width = pair(image_size)

            # what percent of tokens to dropout
            # if int or float given, then assume constant dropout prob
            # otherwise accept a callback that in turn calculates dropout prob from height and width

            self.calc_token_dropout = None

            if callable(token_dropout_prob):
                self.calc_token_dropout = token_dropout_prob

            elif isinstance(token_dropout_prob, (float, int)):
                assert 0. <= token_dropout_prob < 1.
                token_dropout_prob = float(token_dropout_prob)
                self.calc_token_dropout = lambda height, width: token_dropout_prob

            # calculate patching related stuff

            # VG: 3d mod
            assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

            # VG: 3d mod
            patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
            # VG: 3d mod
            patch_dim = channels * (patch_size ** 2)

            self.channels = channels
            self.patch_size = patch_size

            # VG: consider flexifying
            self.to_patch_embedding = nn.Sequential(
                LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                LayerNorm(dim),
            )

            # VG: if statement to allow for this to be replaced with a sin/cos function
            self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
            self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))
            # VG: add a depth
            # VG: also allow for this to be done via concatenation instead

            self.dropout = nn.Dropout(emb_dropout)

            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

            # final attention pooling queries

            self.attn_pool_queries = nn.Parameter(torch.randn(dim))
            self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

            # output to logits

            self.to_latent = nn.Identity()

            self.mlp_head = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, num_classes, bias = False)
            )

        @property
        def device(self):
            return next(self.parameters()).device

        def forward(
            self,
            batched_images: Union[List[Tensor], List[List[Tensor]]], # assume different resolution images already grouped correctly
            group_images = False,
            group_max_seq_len = 2048
        ):
            p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout) and self.training

            arange = partial(torch.arange, device = device)
            pad_sequence = partial(orig_pad_sequence, batch_first = True)

            # auto pack if specified

            if group_images:
                batched_images = group_images_by_max_seq_len(
                    batched_images,
                    patch_size = self.patch_size,
                    calc_token_dropout = self.calc_token_dropout if self.training else None,
                    max_seq_len = group_max_seq_len
                )

            # process images into variable lengthed sequences with attention mask

            num_images = []
            batched_sequences = []
            batched_positions = []
            batched_image_ids = []

            for images in batched_images:
                print("start of batch section with image count of", len(images))
                num_images.append(len(images))

                sequences = []
                positions = []
                image_ids = torch.empty((0,), device = device, dtype = torch.long)

                for image_id, image in enumerate(images):
                    print(image_id, image.shape)
                    # VG: 3d mod
                    assert image.ndim == 3 and image.shape[0] == c
                    # VG: 3d mod
                    image_dims = image.shape[-2:]
                    # VG: 3d mod
                    assert all([divisible_by(dim, p) for dim in image_dims]), f'height and width {image_dims} of images must be divisible by patch size {p}'

                    # VG: 3d mod
                    ph, pw = map(lambda dim: dim // p, image_dims)

                    # VG: 3d mod
                    pos = torch.stack(torch.meshgrid((
                        arange(ph),
                        arange(pw)
                    ), indexing = 'ij'), dim = -1)

                    # VG: 3d mod
                    pos = rearrange(pos, 'h w c -> (h w) c')
                    # VG: 3d mod
                    seq = rearrange(image, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1 = p, p2 = p)

                    seq_len = seq.shape[-2]
                    print("seq_len pre dropout", seq_len)

                    if has_token_dropout: # this will not preserve order b/c of the random selection
                        token_dropout = self.calc_token_dropout(*image_dims)
                        num_keep = max(1, int(seq_len * (1 - token_dropout)))
                        keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                        seq = seq[keep_indices]
                        pos = pos[keep_indices]

                    print("seq_len post dropout", seq.shape[-2])
                    print("pos", pos)
                    image_ids = F.pad(image_ids, (0, seq.shape[-2]), value = image_id)
                    sequences.append(seq)
                    positions.append(pos)

                batched_image_ids.append(image_ids)
                batched_sequences.append(torch.cat(sequences, dim = 0))
                batched_positions.append(torch.cat(positions, dim = 0))

            # derive key padding mask

            lengths = torch.tensor([seq.shape[-2] for seq in batched_sequences], device = device, dtype = torch.long)
            seq_arange = arange(lengths.amax().item())
            key_pad_mask = rearrange(seq_arange, 'n -> 1 n') < rearrange(lengths, 'b -> b 1')
            print("key_pad_mask", key_pad_mask.shape, key_pad_mask)

            # derive attention mask, and combine with key padding mask from above

            batched_image_ids = pad_sequence(batched_image_ids)
            attn_mask = rearrange(batched_image_ids, 'b i -> b 1 i 1') == rearrange(batched_image_ids, 'b j -> b 1 1 j')
            attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j')
            print("attn_mask", attn_mask)

            # combine patched images as well as the patched width / height positions for 2d positional embedding

            patches = pad_sequence(batched_sequences)
            patch_positions = pad_sequence(batched_positions)
            print("batched positions", batched_positions)

            # need to know how many images for final attention pooling

            num_images = torch.tensor(num_images, device = device, dtype = torch.long)

            # to patches

            x = self.to_patch_embedding(patches)

            # factorized 2d absolute positional embedding

            # VG: 3d mod
            h_indices, w_indices = patch_positions.unbind(dim = -1)

            h_pos = self.pos_embed_height[h_indices]
            w_pos = self.pos_embed_width[w_indices]
            # VG: 3d mod
            x = x + h_pos + w_pos

            # embed dropout

            x = self.dropout(x)

            # attention

            x = self.transformer(x, attn_mask = attn_mask)
            print("x after transformer", x.shape)

            # do attention pooling at the end
            print(num_images)
            max_queries = num_images.amax().item()
            print("max_queries", max_queries)

            queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0])
            print("queries", queries.shape)

            # attention pool mask

            image_id_arange = arange(max_queries)

            attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j')
            print("attn_pool_mask", attn_pool_mask.shape, attn_pool_mask)

            attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, 'b j -> b 1 j')

            attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')

            # attention pool

            x = self.attn_pool(queries, context = x, attn_mask = attn_pool_mask) + queries
            print("x after attention pooling", x.shape)

            x = rearrange(x, 'b n d -> (b n) d')

            # each batch element may not have same amount of images

            print("image_id_arange:", image_id_arange)
            print("num_images:", num_images)

            is_images = image_id_arange < rearrange(num_images, 'b -> b 1')
            is_images = rearrange(is_images, 'b n -> (b n)')

            print("is_images", is_images)

            print("x", x.shape)
            x = x[is_images]
            print("x", x.shape)
            # project out to logits

            x = self.to_latent(x)

            print("latent shape:", x.shape)

            return x #self.mlp_head(x)
    return (
        Attention,
        F,
        FeedForward,
        LayerNorm,
        List,
        NaViT,
        RMSNorm,
        Rearrange,
        Transformer,
        Union,
        always,
        default,
        divisible_by,
        exists,
        group_images_by_max_seq_len,
        orig_pad_sequence,
        pair,
        partial,
        rearrange,
        repeat,
    )


@app.cell
def __(NaViT, torch):
    m = NaViT(
        image_size = 256,
        patch_size = 64,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        #token_dropout_prob = 0.1  # token dropout of 10% (keep 90% of tokens)
    )

    # 5 images of different resolutions - List[List[Tensor]]

    # for now, you'll have to correctly place images in same batch element as to not exceed maximum allowed sequence length for self-attention w/ masking

    t = [
        torch.randn(1, 128, 256),
        torch.randn(1, 128, 256),
        torch.randn(1, 256, 128),
        torch.randn(1, 128, 64),
        torch.randn(1, 64, 256)
    ]

    r = m(
        t,
        group_images = True,
        group_max_seq_len = 14
    )
    return m, r, t


@app.cell
def __(L, autoencoder, train_loader):
    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)
    return trainer,


@app.cell
def __(LitAutoEncoder, decoder, encoder, torch):
    # load checkpoint
    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    autoencoder_from_checkpoint = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

    # choose your trained nn.Module
    encoder_from_checkpoint = autoencoder_from_checkpoint.encoder
    encoder.eval()

    # embed 4 fake images!
    fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder_from_checkpoint.device)
    embeddings = encoder_from_checkpoint(fake_image_batch)
    print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
    return (
        autoencoder_from_checkpoint,
        checkpoint,
        embeddings,
        encoder_from_checkpoint,
        fake_image_batch,
    )


if __name__ == "__main__":
    app.run()
