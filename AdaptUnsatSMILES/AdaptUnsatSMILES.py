import random
import re

import networkx as nx
import numpy as np

import SMILESToMol as STMol
# from StructureUnits import *
from StructureUnitsDeNovo import *
from py_common.StructureInformation import SIFromMol


def weighted_choice(weight_list, k=1):
    """
    Select an item from a weighted list using probability-proportional sampling.
    :param weight_list: list of tuple
    :param k: int, number of samples to draw.
    :return: object, the selected item.
    """
    items, weights = zip(*weight_list)
    return random.choices(items, weights=weights, k=k)[0]


def get_unsat_list(monocase_list):
    """
    Get the unsaturation for each monomer string.
    :param monocase_list: list of str. List of monomer representations.
    :return: list of int. Unsaturation for each monomer.
    """
    unsaturation_list = []
    for i in monocase_list:
        try:
            unsat_i = int(i[-6])
            unsaturation_list.append(unsat_i)
        except:
            unsaturation_list.append(0)

    return unsaturation_list


def find_fit_indices(unsaturation_list, monomer_case):
    """
    Identify the possible insertion indices.
    :param unsaturation_list: list of int. List of remaining unsaturation values for each position.
    :param monomer_case: str. SMILES string of the monomer to be inserted.
    :return: list of int.
    """
    bond_type = monomer_case[0]
    required_unsaturation_map = {
        '=': 2,
        '#': 3,
        '≈': 2,
    }
    min_undu_required = required_unsaturation_map.get(bond_type, 1)
    fit_indices = [
        i for i, unsaturation in enumerate(unsaturation_list)
        if unsaturation >= min_undu_required
    ]

    return fit_indices


def _parse_metadata(smiles_unit):
    """
     Parse the internal SMILES monocase.
    :param smiles_unit: str. A SMILES monocase token.
    :return: tuple of int.
    """
    match = re.search(r'!(\d+),(\d+),(\d+)!', smiles_unit)
    if not match:
        raise ValueError(f"Invalid smiles_unit: {smiles_unit}")
    return map(int, match.groups())


def _insert_monocase(target_lst, pos, monocase_lst, as_branch=False, add_dash=False):
    """
    Insert a monocase into a SMILES token list at a specified position.
    :param target_lst: list of str. List of SMILES tokens representing the current structure.
    :param pos: int. Index in 'target_lst' that serves as the anchor position.
    :param monocase_lst: list of str. List of SMILES tokens representing the monocase to be inserted.
    :param as_branch: bool, optional.
                      If True, the monocase is inserted as a branch enclosed by parentheses.
                      If False, the monocase is inserted as a main-chain extension.
                      Default is False.
    :param add_dash: bool, optional.
                     If True, an explicit single bond ('-') is inserted before the monocase.
                     This is typically required for certain aromatic single-bond connections.
                     Default is False.
    :return:
    """
    offset = 1
    if as_branch:
        target_lst.insert(pos + offset, '(')
        offset += 1

    if add_dash:
        monocase_lst[0] = '-' + monocase_lst[0]
        #
        # target_lst.insert(pos + offset, '-')
        # offset += 1

    for i, unit in enumerate(monocase_lst):
        target_lst.insert(pos + offset + i, unit)

    if as_branch:
        target_lst.insert(pos + offset + len(monocase_lst), ')')


def _update_monocase(smiles_unit, delta_unsat, mainchain_connected):
    """
    Update the SMILES token.
    :param smiles_unit: str. A SMILES monocase token.
    :param delta_unsat: int. The amount of unsaturation consumed by the newly formed bond.
    :param mainchain_connected: bool.
                                Flag indicating whether the current bond formation
                                corresponds to a main-chain connection.
    :return: str. The updated SMILES monocase token.
    """
    a, _, d = _parse_metadata(smiles_unit)
    new_a = a - delta_unsat
    new_b = 1 if mainchain_connected else 0
    return re.sub(r'!(.+?)!', f'!{new_a},{new_b},{d}!', smiles_unit)


def _effective_length(smiles_lst):
    return len([x for x in smiles_lst if x not in ('(', ')')])


def _match_aromaticity(old_token, new_token):
    """
    Match the aromaticity (case) of new_token to old_token
    :param old_token: str. Existing SMILES token used as a reference.
    :param new_token: str. New SMILES token.
    :return: str. The updated SMILES token.
    """
    match = re.search(r'[A-Za-z]', old_token)
    if not match:
        return new_token

    old_elem = match.group()
    if old_elem.islower():
        new_token = new_token.lower()
        new_token = '∷' + new_token[1:]
        return new_token
    else:
        return new_token


def _adjust_metadata_for_shared_bond(old_token, new_token):
    """
    Adjust unsaturation or hydrogen count for a shared-bond atom.
    :param old_token: str. Existing SMILES token used as a reference.
    :param new_token: str. New SMILES token.
    :return: str. Adjusted SMILES token.
    """
    old_a, old_b, old_d = _parse_metadata(old_token)
    match = re.search(r'[A-Za-z]', new_token)
    new_elem = match.group()
    if new_elem.islower():
        return re.sub(r'!(.+?)!', f'!0,1,0!', new_token)
    if ('≈' in new_token) or ('≈' in old_token):
        return re.sub(r'!(.+?)!', f'!0,1,0!', new_token)
    return re.sub(r'!(.+?)!', f'!0,1,{old_d}!', new_token)


def _fit_index_index(fit_index_lst, element_index):
    fit_index_index_l = []
    for fit_i in fit_index_lst:
        fit_i_index = element_index.index(fit_i)
        fit_index_index_l.append(fit_i_index)
    return fit_index_index_l


def chemical_rule_atom(SI_mol, element_index, fit_index_lst, radius):
    SF = SI_mol.SF
    Lbon = SI_mol.Lbon
    Latom = np.array(SI_mol.Latom)
    fit_index_index_l = _fit_index_index(fit_index_lst=fit_index_lst, element_index=element_index)
    target_atoms = {'O', 'S', 'N', 'P'}

    de_index=[]
    # relu 1
    for i in fit_index_index_l:
        ssw = np.where(SF[:, i] <= radius)[0]
        if set(Latom[ssw]) & target_atoms:
            de_index.append(element_index[i])

    # relu 2
    for i in fit_index_index_l:
        if (2 in Lbon[i]) or (3 in Lbon[i]):
            de_index.append(element_index[i])

    de_index = list(set(de_index))
    for de_i in de_index:
        fit_index_lst.remove(de_i)
    return fit_index_lst

def chemical_rule_bond(SI_mol, element_index, fit_index_lst):
    SF = SI_mol.SF
    Latom = np.array(SI_mol.Latom)
    fit_index_index_l = _fit_index_index(fit_index_lst=fit_index_lst, element_index=element_index)
    target_atoms = {'O', 'S', 'N', 'P'}

    de_index=[]
    # relu 1
    for i in fit_index_index_l:
        ssw = np.where(SF[:, i] <= 2)[0]
        if set(Latom[ssw]) & target_atoms:
            de_index.append(element_index[i])

    de_index = list(set(de_index))
    for de_i in de_index:
        fit_index_lst.remove(de_i)
    return fit_index_lst

def update_fit_indices(SMILES_R_lst, fit_index_lst, need_add_monocase):
    prefix = need_add_monocase.split('!')[0]
    match_atom = re.search(r'[A-Za-z]+', prefix).group()
    match_atom = match_atom.replace('x', '')

    # update 1: atom 'O', 'S', 'N', 'P'
    if match_atom not in ['O', 'S', 'N', 'P']:
        fit_index_lst = fit_index_lst
    else:
        SMILES_R = ''.join(SMILES_R_lst)
        SMILES_R = re.sub(r'!.*?!', '', SMILES_R)
        SMILES_R = re.sub(r'~', '', SMILES_R)
        SMILES_R = re.sub(r'∷', '', SMILES_R)
        SMILES_R = re.sub(r'≈', '=', SMILES_R)
        mol_infos, element_index = STMol.SMILESToMol(SMILES=SMILES_R)
        SI_mol = SIFromMol(ISI=mol_infos, step_max=None)
        fit_index_lst = chemical_rule_atom(SI_mol=SI_mol, element_index=element_index, fit_index_lst=fit_index_lst, radius=2)

    # update 2: bond
    if ('=' in prefix) or ('#' in prefix):
        SMILES_R = ''.join(SMILES_R_lst)
        SMILES_R = re.sub(r'!.*?!', '', SMILES_R)
        SMILES_R = re.sub(r'~', '', SMILES_R)
        SMILES_R = re.sub(r'∷', '', SMILES_R)
        SMILES_R = re.sub(r'≈', '=', SMILES_R)
        mol_infos, element_index = STMol.SMILESToMol(SMILES=SMILES_R)
        SI_mol = SIFromMol(ISI=mol_infos, step_max=None)
        fit_index_lst = chemical_rule_bond(SI_mol=SI_mol, element_index=element_index, fit_index_lst=fit_index_lst)

    return fit_index_lst

def get_graph_hash(SMILES):
    single_element = STMol.get_str_list(SMILES=SMILES)
    info_dict = STMol.get_info(single_element=single_element)
    connect_info = STMol.get_adj(single_element=single_element, info_dict=info_dict)
    bond_list = STMol.get_L_bond(info_dict=info_dict, connect_info=connect_info)
    atom_info = info_dict['atom_info']
    atom_info_m = np.matrix(atom_info, dtype=object)
    Latom = atom_info_m[:, 1].flatten().tolist()[0]

    G = nx.Graph()
    for i, atom_idx in enumerate(Latom):
        G.add_node(i, atomic_num=atom_idx)

    for i in range(len(connect_info)):
        for j in range(len(connect_info[i])):
            neighbor = connect_info[i][j]
            bond_order = bond_list[i][j]
            if i < neighbor:
                G.add_edge(i, neighbor, bond_order=bond_order)

    SMILES_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr='atomic_num', edge_attr='bond_order')
    return SMILES_hash


def modif_R(SMILES_R_lst, fit_index_lst, need_add_monocase, prob):
    """
    Modify the R-site SMILES token list by inserting a new monocase at a randomly selected valid position.
    :param SMILES_R_lst: list of str.  List of SMILES tokens.
    :param fit_index_lst: list of int.
           Indices of SMILES_R_lst where insertion of a new monocase is chemically and topologically allowed.
    :param need_add_monocase: str. SMILES string of the monocase to be inserted.
    :param prob: float.
                 Probability of forming a shared bond between two adjacent cyc
                 sites when such a configuration is allowed.
    :return: list of str. The updated SMILES_R_lst after in-place modification.
    """
    need_add_monocase_L = need_add_monocase.split('+')
    bond_type = need_add_monocase[0]
    x = random.choice(fit_index_lst)

    curr_unit = SMILES_R_lst[x]
    curr_bond = curr_unit[0]

    is_shared_bond = False
    special_bond = {'~', '∷'}
    if (
            curr_bond in special_bond
            and bond_type in special_bond
            and (x + 1) in fit_index_lst
            and random.random() < prob
    ):
        try:
            L_token=need_add_monocase_L[2]
            R_token=SMILES_R_lst[x + 2]
            is_shared_bond = not (
                    ('≈' in L_token and '≈' in R_token) or
                    ('∷' in L_token and '≈' in R_token) or
                    ('≈' in L_token and '∷' in R_token)
            )
        except:
            is_shared_bond = False

    if is_shared_bond:
        mm = curr_unit.index('!') - 1
        nn = SMILES_R_lst[x + 1].index('!') - 1
        if curr_unit[mm].isdigit() or SMILES_R_lst[x + 1][nn].isdigit():
            return SMILES_R_lst

        new0 = _match_aromaticity(SMILES_R_lst[x], need_add_monocase_L[0])
        new1 = _match_aromaticity(SMILES_R_lst[x + 1], need_add_monocase_L[1])
        new0 = _adjust_metadata_for_shared_bond(SMILES_R_lst[x], new0)
        if '≈' in SMILES_R_lst[x + 1]:
            new0 = re.sub(r'!(.+?)!', f'!0,1,0!', new0)
        if '≈' in need_add_monocase_L[1]:
            new0 = re.sub(r'!(.+?)!', f'!0,1,0!', new0)
        new1 = _adjust_metadata_for_shared_bond(SMILES_R_lst[x + 1], new1)
        if '≈' in SMILES_R_lst[x + 2]:
            new1 = re.sub(r'!(.+?)!', f'!0,1,0!', new1)
        if '≈' in need_add_monocase_L[2]:
            new1 = re.sub(r'!(.+?)!', f'!0,1,0!', new1)

        SMILES_R_lst[x:x + 2] = [new0, new1]

        SMILES_R_lst.insert(x + 2, '(')

        for i in range(2, len(need_add_monocase_L)):
            SMILES_R_lst.insert(x + 1 + i, need_add_monocase_L[i])
        SMILES_R_lst.insert(x + 1 + len(need_add_monocase_L), ')')

        return SMILES_R_lst

    a, b, d = _parse_metadata(curr_unit)
    first_insert = (b == 0)
    aromatic_single = (
            curr_bond in special_bond
            and curr_unit[1].islower()
            and bond_type in special_bond
            and need_add_monocase[1].islower()
    )
    _insert_monocase(
        SMILES_R_lst,
        x,
        need_add_monocase_L,
        as_branch=not first_insert,
        add_dash=aromatic_single,
    )
    delta_unsat = {'=': 2, '#': 3}.get(bond_type, 1)
    SMILES_R_lst[x] = _update_monocase(curr_unit, delta_unsat, mainchain_connected=True)

    return SMILES_R_lst


# ============================================================================================
def produce_SMILES(N1, N2, cyc_max=0, shared_bond_prob=0.05, use_scaffold=False):
    """
    Produce SMILES
    :param N1: int. Number of substructures.
    :param N2: int. Number of heavy atoms.
    :param cyc_max: int. The maximum number of cycles in a scaffold.
    :param shared_bond_prob: float, 0-1. The probability of shared bond.
    :param use_scaffold: bool. Whether to use scaffold.
    :return: list of str. The list of generated SMILES.
    """
    SMILES_R_all = []
    Hash_temp = []
    # ---------- initialization ----------
    while len(SMILES_R_all) < N1:
        if use_scaffold:
            first_monocase = weighted_choice(weight_list=first_monocase_R, k=1)
        else:
            first_monocase = weighted_choice(weight_list=first_monocase_de, k=1)
        SMILES_R_lst = [first_monocase]
        ring_index = cyc_max + 1

        current_len = _effective_length(SMILES_R_lst)

        # ---------- growth loop ----------
        while current_len < N2:
            unsat_list = get_unsat_list(SMILES_R_lst)
            if set(unsat_list) == {0}:
                SMILES_R_lst = []
                break

            # ---- select element and unsaturation ----
            try_num = 0
            while try_num < 10:
                element = weighted_choice(weight_list=element_num, k=1)
                valid_unsat = [u for u in unsat_list if u != 0]
                chosen_unsat = int(random.choice(valid_unsat))

                element_unsat_dict = element_dict[f'{element}_nums']
                max_unsat = max(element_unsat_dict, key=int)
                chosen_unsat = str(min(chosen_unsat, int(max_unsat)))

                monocase_candidates = element_dict[f'{element}_nums'][chosen_unsat]
                monocase = weighted_choice(weight_list=monocase_candidates, k=1)

                # ---- ring handling ----
                if element in cyc_l:
                    if ring_index < 10:
                        monocase = monocase.replace('x', f'{ring_index}')
                    else:
                        monocase = monocase.replace('x', f'%{ring_index}')
                    ring_index += 1

                # ---- insertion ----
                fit_index_lst_ini = find_fit_indices(unsaturation_list=unsat_list, monomer_case=monocase)
                fit_index_lst = update_fit_indices(SMILES_R_lst=SMILES_R_lst, fit_index_lst=fit_index_lst_ini,
                                                   need_add_monocase=monocase)
                if len(fit_index_lst) == 0:
                    try_num += 1
                else:
                    break
            if len(fit_index_lst) == 0:
                break
            else:
                SMILES_R_lst = modif_R(SMILES_R_lst=SMILES_R_lst,
                                       fit_index_lst=fit_index_lst,
                                       need_add_monocase=monocase,
                                       prob=shared_bond_prob)
                current_len = _effective_length(SMILES_R_lst)

        # ---------- hydrogen completion ----------
        if current_len == N2:
            for i, token in enumerate(SMILES_R_lst):
                if len(token) <= 1:
                    continue

                H_total = int(token[-6]) + int(token[-2])
                if H_total <= 0:
                    SMILES_R_lst[i] = token[:-7]
                    continue

                try:
                    if SMILES_R_lst[i + 1] == ')':
                        SMILES_R_lst[i] = token[:-7] + '([H])' * (H_total - 1) + '[H]'
                    else:
                        SMILES_R_lst[i] = token[:-7] + '([H])' * H_total
                except IndexError:
                    SMILES_R_lst[i] = SMILES_R_lst[i][:-7] + '([H])' * (H_total - 1) + '[H]'

            SMILES_R = ''.join(SMILES_R_lst)
            SMILES_R = re.sub(r'~', '', SMILES_R)
            SMILES_R = re.sub(r'∷', '', SMILES_R)
            SMILES_R = re.sub(r'≈', '=', SMILES_R)

            # Deduplicate by hash
            SMILES_hash = get_graph_hash(SMILES=SMILES_R)

            if SMILES_hash not in Hash_temp:
                SMILES_R_all.append(SMILES_R)
                Hash_temp.append(SMILES_hash)

    return SMILES_R_all


