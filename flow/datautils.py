import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from fast_jtnn import *
from fast_molvae import *
import pickle as pickle
import os, random
from tqdm import tqdm

class MolTreeWithPropertyFolder(object):
    def __init__(
        self,
        mol_folder,
        property_folder,
        vocab,
        batch_size,
        num_workers=4,
        shuffle=True,
        assm=True,
        replicate=None,
        with_property=False,
    ):

        self.mol_folder = mol_folder
        self.property_folder = property_folder
        self.data_files = [fn for fn in os.listdir(mol_folder)]
        self.data_files.sort(key=lambda k: int(k[:-4].split("-")[-1]) )

        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            mf = os.path.join(self.mol_folder, fn)
            pf = os.path.join(self.property_folder, fn)

            with open(mf, "rb") as f:
                mol_data = pickle.load(f)
            with open(pf, "rb") as f:
                property_data = pickle.load(f)

            if self.shuffle:
                zipped = list(zip(mol_data, property_data))
                random.shuffle(zipped)
                mol_data, property_data = zip(*zipped)

            mol_batches = [
                mol_data[i : i + self.batch_size]
                for i in range(0, len(mol_data), self.batch_size)
            ]
            if len(mol_batches[-1]) < self.batch_size:
                mol_batches.pop()

            property_batches = [
                property_data[i : i + self.batch_size]
                for i in range(0, len(property_data), self.batch_size)
            ]
            if len(property_batches[-1]) < self.batch_size:
                property_batches.pop()

            dataset = MolTreeWithPropertyDataset(
                mol_batches, property_batches, self.vocab, self.assm
            )
            dataloader = DataLoader(
                dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
            )  # , num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del (
                mol_data,
                property_data,
                mol_batches,
                property_batches,
                dataset,
                dataloader,
            )


class MolTreeWithPropertyDataset(Dataset):
    def __init__(self, mol_data, property_data, vocab, assm=True):
        self.mol_data = mol_data
        self.property_data = property_data
        self.vocab = vocab
        self.assm = assm

        assert len(self.mol_data) == len(self.property_data)

    def __len__(self):
        return len(self.mol_data)

    def __getitem__(self, idx):
        return (
            tensorize(self.mol_data[idx], self.vocab, assm=self.assm),
            torch.tensor(self.property_data[idx]).unsqueeze(1),
        )


def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i, mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            # Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1:
                continue
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx)


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


class FlowDataset(Dataset):
    def __init__(self, mol_path, property_path, vocab, model, save_path, load=False):
        sufix = property_path.split("/")[-2] if property_path.endswith("/") else property_path.split("/")[-1]
        if load:
            self.W_tree = torch.load(f"{save_path}/W_tree_{sufix}.pt")
            self.W_mol = torch.load(f"{save_path}/W_mol_{sufix}.pt")
            self.A = torch.load(f"{save_path}/A_{sufix}.pt")
            print(self.W_tree.shape)
        else:
            W_tree, W_mol, A = list(), list(), list()
            loader = MolTreeWithPropertyFolder(mol_path, property_path, vocab, batch_size=32)
            model.eval()
            with torch.no_grad():
                for batch in tqdm(loader):
                    x_batch, a_batch = batch
                    x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
                    x_tree_vecs, _, x_mol_vecs = model.encode(x_jtenc_holder, x_mpn_holder)
                    W_tree += [model.T_mean(x_tree_vecs)]
                    W_mol += [model.G_mean(x_mol_vecs)]
                    A += [a_batch]

            del loader
            self.W_tree = torch.cat(W_tree)
            self.W_mol = torch.cat(W_mol)
            self.A = torch.cat(A)
            torch.save(self.W_tree, f"{save_path}/W_tree_{sufix}.pt")
            torch.save(self.W_mol, f"{save_path}/W_mol_{sufix}.pt")
            torch.save(self.A, f"{save_path}/A_{sufix}.pt")

    def __len__(self):
        return len(self.W_tree)

    def __getitem__(self, idx):
        return self.W_tree[idx], self.W_mol[idx], self.A[idx]
