import rdkit
from rdkit.Chem import Crippen, QED
from sascorer import calculateScore


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
