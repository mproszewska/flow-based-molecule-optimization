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
        smiles_folder,
        mol_folder,
        scaffold_folder,
        attr_folder,
        vocab,
        batch_size,
        num_workers=4,
        shuffle=False,
        assm=True,
        replicate=None,
        with_attr=False,
    ):
        self.smiles_folder = smiles_folder
        self.mol_folder = mol_folder
        self.scaffold_folder = scaffold_folder
        self.attr_folder = attr_folder
        self.data_files = [fn for fn in os.listdir(mol_folder)]
        self.data_files.sort(key=lambda k: int(k[:-4].split("-")[-1]))

        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            sf = os.path.join(self.smiles_folder, fn)
            mf = os.path.join(self.mol_folder, fn)
            scf = os.path.join(self.scaffold_folder, fn)
            pf = os.path.join(self.attr_folder, fn)

            with open(sf, "rb") as f:
                smiles_data = pickle.load(f)
            with open(mf, "rb") as f:
                mol_data = pickle.load(f)
            with open(scf, "rb") as f:
                scaffold_data = pickle.load(f)
            with open(pf, "rb") as f:
                attr_data = pickle.load(f)

            if self.shuffle:
                zipped = list(zip(smiles_data, mol_data, scaffold_data, attr_data))
                random.shuffle(zipped)
                smiles_data, mol_data, scaffold_data, attr_data = zip(*zipped)

            smiles_batches = [
                smiles_data[i : i + self.batch_size]
                for i in range(0, len(smiles_data), self.batch_size)
            ]
            if len(smiles_batches[-1]) < self.batch_size:
                smiles_batches.pop()

            mol_batches = [
                mol_data[i : i + self.batch_size]
                for i in range(0, len(mol_data), self.batch_size)
            ]
            if len(mol_batches[-1]) < self.batch_size:
                mol_batches.pop()

            scaffold_batches = [
                scaffold_data[i : i + self.batch_size]
                for i in range(0, len(scaffold_data), self.batch_size)
            ]
            if len(scaffold_batches[-1]) < self.batch_size:
                scaffold_batches.pop()

            attr_batches = [
                attr_data[i : i + self.batch_size]
                for i in range(0, len(attr_data), self.batch_size)
            ]
            if len(attr_batches[-1]) < self.batch_size:
                attr_batches.pop()

            dataset = MolTreeWithPropertyDataset(
                smiles_batches,
                mol_batches,
                scaffold_batches,
                attr_batches,
                self.vocab,
                self.assm,
            )
            dataloader = DataLoader(
                dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
            )  # , num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del (
                smiles_data,
                mol_data,
                scaffold_data,
                attr_data,
                smiles_batches,
                mol_batches,
                attr_batches,
                dataset,
                dataloader,
            )


class MolTreeWithPropertyDataset(Dataset):
    def __init__(
        self, smiles_data, mol_data, scaffold_data, attr_data, vocab, assm=True
    ):
        self.smiles_data = smiles_data
        self.mol_data = mol_data
        self.scaffold_data = scaffold_data
        self.attr_data = attr_data
        self.vocab = vocab
        self.assm = assm

        assert len(self.mol_data) == len(self.attr_data)

    def __len__(self):
        return len(self.mol_data)

    def __getitem__(self, idx):
        return (
            self.smiles_data[idx],
            None,  # tensorize(self.mol_data[idx], self.vocab, assm=self.assm),
            self.scaffold_data[idx],
            torch.tensor(self.attr_data[idx]).unsqueeze(1),
        )


def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles if tree is not None else None for tree in tree_batch]

    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    # if jtenc_holder is None: return None
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

    if len(cands) == 0:
        return None
    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx)


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        # if mol_tree is None: continue
        for node in mol_tree.nodes:
            node.idx = tot
            """
            if not vocab.exists(node.smiles):
                print("Node not in vocab")
                node.wid = -1
                continue
            """
            node.wid = vocab.get_index(node.smiles)
            tot += 1


class FlowDataset(Dataset):
    def __init__(
        self,
        smiles_path,
        mol_path,
        scaffold_path,
        attr_path,
        vocab,
        model,
        save_path,
        use_logvar=False,
        load=False,
    ):
        sufix = (
            attr_path.split("/")[-2]
            if attr_path.endswith("/")
            else attr_path.split("/")[-1]
        )
        self.use_logvar = use_logvar
        if load:
            device = next(model.parameters()).device
            self.S = torch.load(f"{save_path}/S.pt", map_location=device)
            self.W_tree = torch.load(
                f"{save_path}/W_tree.pt", map_location=device
            ).float()
            self.W_mol = torch.load(
                f"{save_path}/W_mol.pt", map_location=device
            ).float()
            self.Sc = torch.load(f"{save_path}/Sc.pt", map_location=device).float()
            self.A = torch.load(
                f"{save_path}/A_{sufix}.pt", map_location=device
            ).float()

            if use_logvar:
                self.W_tree_logvar = torch.load(
                    f"{save_path}/W_tree_logvar.pt", map_location=device
                ).float()
                self.W_mol_logvar = torch.load(
                    f"{save_path}/W_mol_logvar.pt", map_location=device
                ).float()
        else:
            S, W_tree, W_mol, Sc, A = list(), list(), list(), list(), list()
            if use_logvar:
                W_tree_logvar, W_mol_logvar = list(), list()
            loader = MolTreeWithPropertyFolder(
                smiles_path, mol_path, scaffold_path, attr_path, vocab, batch_size=32
            )
            iterator = iter(loader)
            model.eval()
            with torch.no_grad():
                for batch in tqdm(loader):

                    s_batch, x_batch, sc_batch, a_batch = batch
                    x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
                    x_tree_vecs, _, x_mol_vecs = model.encode(
                        x_jtenc_holder, x_mpn_holder
                    )
                    S += [s_batch]
                    W_tree += [model.T_mean(x_tree_vecs)]
                    W_mol += [model.G_mean(x_mol_vecs)]
                    Sc += [sc_batch]
                    A += [a_batch]
                    if use_logvar:
                        W_tree_logvar += [model.T_var(x_tree_vecs)]
                        W_mol_logvar += [model.G_var(x_mol_vecs)]
            del loader
            self.S = np.concatenate(S)
            self.W_tree = torch.cat(W_tree)
            self.W_mol = torch.cat(W_mol)
            self.Sc = torch.from_numpy(np.concatenate(Sc)).float().unsqueeze(-1)
            self.A = torch.cat(A)
            torch.save(self.W_tree, f"{save_path}/W_tree.pt")
            torch.save(self.W_mol, f"{save_path}/W_mol.pt")
            torch.save(self.Sc, f"{save_path}/Sc.pt")
            torch.save(self.A, f"{save_path}/A_{sufix}.pt")
            torch.save(self.S, f"{save_path}/S.pt")
            if use_logvar:
                self.W_tree_logvar = torch.cat(W_tree_logvar)
                self.W_mol_logvar = torch.cat(W_mol_logvar)
                torch.save(self.W_tree_logvar, f"{save_path}/W_tree_logvar.pt")
                torch.save(self.W_mol_logvar, f"{save_path}/W_mol_logvar.pt")
        if self.A.shape[-1] == 1:
            self.A[self.A >= 100] = np.nan
            print(len(self.A))
            idx = (1 - np.isnan(self.A.cpu())[:, 0]).bool()
            self.S, self.W_tree, self.W_mol, self.Sc, self.A = (
                self.S[idx],
                self.W_tree[idx],
                self.W_mol[idx],
                self.Sc[idx],
                self.A[idx],
            )
            print(len(self.A))
            if use_logvar:
                (self.W_tree_logvar, self.W_mol_logvar,) = (
                    self.W_tree_logvar[idx],
                    self.W_mol_logvar[idx],
                )

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        if self.use_logvar:
            return (
                self.S[idx],
                (self.W_tree[idx], self.W_tree_logvar[idx]),
                (self.W_mol[idx], self.W_mol_logvar[idx]),
                self.Sc[idx],
                self.A[idx],
            )
        else:
            return (
                self.S[idx],
                self.W_tree[idx],
                self.W_mol[idx],
                self.Sc[idx],
                self.A[idx],
            )
