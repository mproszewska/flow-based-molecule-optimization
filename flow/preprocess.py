import sys

sys.path.append("../")
import torch
import torch.nn as nn
from multiprocessing import Pool
import numpy as np
import os
from tqdm import tqdm

import math, random, sys
from argparse import ArgumentParser
import pickle
import pandas as pd

from fast_jtnn import *
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


def tensorize(smiles, assm=True):
    print(smiles, smiles.__class__)
    if isinstance(smiles, float) and np.isnan(smiles):
        smiles = ""
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree


def convert(train_path, properties, pool, num_splits, output_path):
    print("Converting smiles with properties: ", properties)

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    out_path = os.path.join(output_path, "./")
    if os.path.isdir(out_path) is False:
        os.makedirs(out_path)
    for subdir in ["mol", "scaffold_one_hot", "smiles", "fingerprint_1024"] + properties:
        subdir_path = os.path.join(out_path, subdir)
        if os.path.isdir(subdir_path) is False:
            os.makedirs(subdir_path)

    df = pd.read_csv(train_path)
    smiles = df["smiles"].tolist()
    print("Tensorizing smiles.....", df.shape)
    smiles_data = pool.map(tensorize, smiles)
    smiles_data_split = np.array_split(smiles_data, num_splits)

    for split_id in tqdm(range(num_splits)):
        with open(
            os.path.join(output_path, "mol/tensors-%d.pkl" % split_id), "wb"
        ) as f:
            pickle.dump(smiles_data_split[split_id], f)
    
    print("Tensorizing scaffolds.....")
    scaffold_data = df["scaffold_one_hot"].to_numpy()
    scaffold_data_split = np.array_split(scaffold_data, num_splits)

    for split_id in tqdm(range(num_splits)):
        with open(
            os.path.join(output_path, "scaffold_one_hot/tensors-%d.pkl" % split_id),
            "wb",
        ) as f:
            pickle.dump(scaffold_data_split[split_id], f)

    print("Tensorizing smiles strings.....")
    smiles_split = np.array_split(smiles, num_splits)
    for split_id in tqdm(range(num_splits)):
        with open(
            os.path.join(output_path, "smiles/tensors-%d.pkl" % (split_id)),
            "wb",
        ) as f:
            pickle.dump(smiles_split[split_id], f)
    
    print("Extracting and tensorizing fingerprints.....")
    fingerprint_data = np.empty((len(smiles), 1024), dtype=np.int8)
    for i, s in enumerate(smiles):
        #fp = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s))
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024)
        arr = np.zeros((0,), dtype=np.int8) 
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprint_data[i] = arr
    fingerprint_data_split = np.array_split(fingerprint_data, num_splits)

    for split_id in tqdm(range(num_splits)):
        with open(
            os.path.join(output_path, "fingerprint_1024/tensors-%d.pkl" % split_id),
            "wb",
        ) as f:
            pickle.dump(fingerprint_data_split[split_id], f)

    print("Tensorizing properties.....")
    for property in properties:
        property_data = df[property].to_numpy()
        property_data_split = np.array_split(property_data, num_splits)

        for split_id in tqdm(range(num_splits)):
            with open(
                os.path.join(output_path, "%s/tensors-%d.pkl" % (property, split_id)),
                "wb",
            ) as f:
                pickle.dump(property_data_split[split_id], f)

    print("Tensorizing Complete")

    return True


def main_preprocess(
    train_path, output_path, property, num_splits=10, njobs=os.cpu_count()
):
    pool = Pool(njobs)
    convert(train_path, property, pool, num_splits, output_path)
    return True


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = ArgumentParser()
    parser.add_argument("-t", "--train", type=str, required=True)
    parser.add_argument("-p", "--properties", nargs="*", type=str, required=True)
    parser.add_argument("-n", "--split", default=10, type=int)
    parser.add_argument("-j", "--jobs", default=8, type=int)
    parser.add_argument("-o", "--output", type=str, required=True)

    args = parser.parse_args()

    pool = Pool(args.jobs)
    convert(args.train, args.properties, pool, args.split, args.output)
