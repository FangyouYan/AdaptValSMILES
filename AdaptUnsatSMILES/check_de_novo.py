import json
from rdkit import Chem


def standardize_SMILES(SMILES):
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        print(f'Error:{SMILES}')
        stand_SMILES = SMILES
    else:
        stand_SMILES = Chem.MolToSmiles(mol,canonical=True)

    return stand_SMILES


with open('./de_novo_design_test/De_SMILES.json','r', encoding='utf-8') as f:
    SMILES_R = json.load(f)

SMILES_R_keys = list(SMILES_R.keys())


for key in SMILES_R_keys:
    temp_list = []
    rep_num = 0
    for R_i in range(len(SMILES_R[key])):
        cation_SMILES_i = SMILES_R[key][R_i]
        stand_SMILES_i = standardize_SMILES(cation_SMILES_i)
        if stand_SMILES_i not in temp_list:
            temp_list.append(stand_SMILES_i)
        else:
            repet_index = temp_list.index(stand_SMILES_i)
            temp_list.append(stand_SMILES_i)
            rep_num += 1
            print(f'repetitive:{key}_{R_i}_{repet_index}:::{cation_SMILES_i}')
    print(f'---------------------{key}-----------{rep_num}----------------------------------------------------')