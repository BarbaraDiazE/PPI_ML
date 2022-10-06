from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def draw_one_molecule(mol_smiles):
    return


def get_ECFP4_bits(mol_smiles: str):
    molecule = Chem.MolFromSmiles(mol_smiles)
    bi = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, radius=2, bitInfo=bi)
    return list(fp.GetOnBits())


def get_ECFP6_bits(mol_smiles: str):
    molecule = Chem.MolFromSmiles(mol_smiles)
    bi = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, radius=3, bitInfo=bi)
    return list(fp.GetOnBits())


def draw_one_fragment(mol_smiles: str, fragment, r: int):
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import Draw

    bi = {}
    molecule = Chem.MolFromSmiles(mol_smiles)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, radius=r, bitInfo=bi)
    return Draw.DrawMorganBit(molecule, fragment, bi)
