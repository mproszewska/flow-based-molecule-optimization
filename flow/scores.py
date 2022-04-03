import numpy as np
import rdkit
from rdkit.Chem import Crippen, QED
from sascorer import calculateScore
from tqdm import tqdm
from rdkit.Chem.Lipinski import NumAromaticRings


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
        rdkit.Chem.RDKFingerprint(rdkit.Chem.MolFromSmiles(s)),
        rdkit.Chem.RDKFingerprint(rdkit.Chem.MolFromSmiles(t)),
    )


def calculate_diversity(smiles):
    similarities = list()
    smiles_unique = set([rdkit.Chem.CanonSmiles(s) for s in smiles])
    return len(smiles_unique) / len(smiles)
    """
    smiles = [rdkit.Chem.RDKFingerprint(rdkit.Chem.MolFromSmiles(s)) for s in smiles]
    
    for i, s in tqdm(enumerate(smiles), total=len(smiles)):
        for t in smiles[i + 1 :]:
            similarities += [rdkit.DataStructs.FingerprintSimilarity(s, t)]
    return (np.array(similarities) < 0.5).mean()
    """


def calculate_aromatic_rings(smile):
    mol = rdkit.Chem.MolFromSmiles(smile)
    return NumAromaticRings(mol)
