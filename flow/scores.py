import numpy as np
import rdkit
from rdkit import DataStructs
from rdkit.Chem import AllChem, Crippen, MolFromSmiles, QED
from sascorer import calculateScore
from tqdm import tqdm
from rdkit.Chem.Lipinski import NumAromaticRings
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem import MACCSkeys

def calculate_sas(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    return calculateScore(mol)


def calculate_qed(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    return QED.qed(mol)


def calculate_logP(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    logP = Crippen.MolLogP(mol)
    return logP


def calculate_similarity(s, t):
    return rdkit.DataStructs.FingerprintSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(s), 2),
        AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(t), 2)
        #rdkit.Chem.RDKFingerprint(rdkit.Chem.MolFromSmiles(s)),
        #rdkit.Chem.RDKFingerprint(rdkit.Chem.MolFromSmiles(t)),
    )


def calculate_diversity(smiles):
    similarities = list()
    smiles_unique = set([rdkit.Chem.CanonSmiles(s) for s in smiles])
    return len(smiles_unique) / len(smiles)


def calculate_aromatic_rings(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    return NumAromaticRings(mol)


def calculate_scaffold_code(smiles, scaffold_dict):
    scaffold = rdkit.Chem.CanonSmiles(MurckoScaffoldSmiles(smiles))
    return scaffold_dict[scaffold] if scaffold in scaffold_dict else len(scaffold_dict)


def calculate_fingerprint(smiles):
    fp = AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(smiles), 2, 1024)   
    #fp = MACCSkeys.GenMACCSKeys(MolFromSmiles(smiles)) 
    arr = np.zeros((0,), dtype=np.int8) 
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
