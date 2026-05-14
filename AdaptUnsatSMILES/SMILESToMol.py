import copy
import datetime

import numpy as np


aliphatic_organic = ['H', 'B', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
aromatic_organic = ['b', 'c', 'n', 'o', 's', 'p']

bond = ['-', '=', '#', '∷']
ring_digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '%']


def get_str_list(SMILES):
    i = 0
    single_element = []
    while i < len(SMILES):
        # print(single_element)
        if SMILES[i:i + 2] in aliphatic_organic:
            single_element.append(SMILES[i:i + 2])
            i += 2
        elif SMILES[i] in aliphatic_organic:
            x = SMILES[i]
            j = 1
            while j < min(len(SMILES[i:]), 15):
                if SMILES[i + j] in ring_digit:
                    x += SMILES[i + j]
                    j += 1
                else:
                    break
            single_element.append(x)
            i += len(x)
        elif SMILES[i] in aromatic_organic:
            x = SMILES[i]
            j = 1
            while j < min(len(SMILES[i:]), 15):
                if SMILES[i + j] in ring_digit:
                    x += SMILES[i + j]
                    j += 1
                else:
                    break
            single_element.append(x)
            i += len(x)
        elif SMILES[i] in '().':
            single_element.append(SMILES[i])
            i += 1
        elif SMILES[i] in bond:
            if SMILES[i + 1] == '[':
                x = SMILES[i + 1:].find(']') + (i + 1)
                y = SMILES[i:x + 1]
                j = 1
                while j < min(len(SMILES[x:]), 15):
                    if SMILES[x + j] in ring_digit:
                        y += SMILES[x + j]
                        j += 1
                    else:
                        break
                single_element.append(y)
                i += len(y)
            else:
                x = SMILES[i:i + 2]
                j = 1
                while j < min(len(SMILES[i + 1:]), 15):
                    if SMILES[i + 1 + j] in ring_digit:
                        x += SMILES[i + 1 + j]
                        j += 1
                    else:
                        break
                single_element.append(x)
                i += len(x)
        elif SMILES[i] == '[':
            x = SMILES[i:].find(']') + i
            y = SMILES[i:x + 1]
            j = 1
            while j < min(len(SMILES[x:]), 15):
                if SMILES[x + j] in ring_digit:
                    y += SMILES[x + j]
                    j += 1
                else:
                    break
            single_element.append(y)
            i += len(y)

    return single_element


def get_info(single_element):
    single_element_f = copy.deepcopy(single_element)
    info_dict = {}
    left_index = []
    right_index = []
    dot_index = []
    atom_info = []

    for i in range(len(single_element_f)):
        if single_element_f[i] == '(':
            left_index.append(i)
        elif single_element_f[i] == ')':
            right_index.append(i)
        elif single_element_f[i] == '.':
            dot_index.append(i)
        else:
            x = single_element_f[i]
            if len(x) == 1:
                atom_info.append([i, x, '', '', '', []])
            else:
                y1, y2, y3, y4, y5 = '', '', '', '', []
                for j in range(len(x)):
                    if x[j].isalpha():
                        y1 += x[j]
                atom_info.append([i, y1, y2, y3, y4, y5])


    for i in range(len(atom_info)):
        x0 = atom_info[i][0]
        x1 = atom_info[i][1]
        y = single_element_f[x0]
        if len(y) > 1:
            if y[0] in bond:
                atom_info[i][2] = y[0]
            if '[' in y:
                yy1 = y.find('[')
                yy2 = y.find(']')
                y = y[yy1:yy2 + 1]
                y = y.split(x1)
                y30, y40 = '', ''
                y3 = [y30 + j for j in y[0] if j != '[']
                y4 = [y40 + j for j in y[1] if j != ']']
                atom_info[i][3] = ''.join(y3)
                atom_info[i][4] = ''.join(y4)


    for i in range(len(atom_info)):
        x0 = atom_info[i][0]
        x1 = atom_info[i][1]
        y = single_element_f[x0]
        ring_num = ''
        if len(y) > 1:
            if '[' in y:
                yy2 = y.find(']')
                ring_num = y[yy2 + 1:]
            else:
                ring_num = y.split(x1)[-1]
        if len(ring_num) > 0:
            if '%' in ring_num:
                ring2 = ring_num.find('%')
                ring_num1 = ring_num[:ring2]
                ring_num2 = ring_num[ring2:]
                ring_num2 = ring_num2.split('%')[1:]
                if len(ring_num1) == 0:
                    atom_info[i][5] = ring_num2
                else:
                    atom_info[i][5] = list(ring_num1) + ring_num2
            else:
                atom_info[i][5] = list(ring_num)

    info_dict['left_index'] = left_index
    info_dict['right_index'] = right_index
    info_dict['dot_index'] = dot_index
    info_dict['atom_info'] = atom_info
    return info_dict


def get_adj(single_element, info_dict):

    left_index = info_dict['left_index']
    right_index = info_dict['right_index']
    atom_info = info_dict['atom_info']
    atom_info_m = np.matrix(atom_info, dtype=object)
    atom_index = atom_info_m[:, 0].flatten().tolist()[0]
    ring_info = atom_info_m[:, 5].flatten().tolist()[0]
    connect_list = [[] for i in atom_index]

    for i in range(1, len(atom_index)):
        a = single_element[atom_index[i - 1]:atom_index[i]]
        if (')' not in a) & ('.' not in a):
            connect_list[i - 1].append(i)
            connect_list[i].append(i - 1)

    right_index_f = copy.deepcopy(right_index)
    left_index_f = copy.deepcopy(left_index)
    flag = 1
    while right_index_f:
        right_index_i = right_index_f[0]
        diff = [right_index_i - j for j in left_index_f]
        min_diff = float('inf')
        for j in range(len(diff)):
            if 0 < diff[j] <= min_diff:
                min_diff = diff[j]
        min_diff_index = diff.index(min_diff)
        left_index_i = left_index_f[min_diff_index]

        if '.' in single_element[left_index_i:right_index_i]:
            flag = 0
            info_1 = 'SMILES，(.) !'
            break

        x1 = atom_index.index(left_index_i - 1)
        right_index_i = right_index_f[0]
        if (right_index_i + 1) not in left_index_f:
            x2 = atom_index.index(right_index_i + 1)
            left_index_f.remove(left_index_i)
            right_index_f.remove(right_index_i)
        else:
            x2 = atom_index.index(right_index_i + 2)
            left_index_f.remove(right_index_i + 1)
            right_index_f.remove(right_index_i)
        connect_list[x1].append(x2)
        connect_list[x2].append(x1)

    ring_info_f = copy.deepcopy(ring_info)
    for i in range(len(ring_info_f)):
        if len(ring_info_f[i]) > 0:
            while ring_info_f[i]:
                x = ring_info_f[i][0]
                for j in range(i + 1, len(ring_info_f)):
                    if x in ring_info_f[j]:
                        connect_list[i].append(j)
                        connect_list[j].append(i)
                        ring_info_f[i].remove(x)
                        ring_info_f[j].remove(x)
                        break
    if flag == 1:
        return connect_list
    else:
        return info_1


def get_L_bond(info_dict, connect_info):

    atom_info = info_dict['atom_info']
    atom_info_m = np.matrix(atom_info, dtype=object)
    element_info = atom_info_m[:, 1].flatten().tolist()[0]
    bond_info = atom_info_m[:, 2].flatten().tolist()[0]
    bond_list = [['' for j in range(len(connect_info[i]))] for i in range(len(connect_info))]

    for i in range(len(connect_info)):
        for j in range(len(connect_info[i])):
            if (element_info[i].islower()) & (element_info[connect_info[i][j]].islower()):
                bond_list[i][j] = 1.5
            else:
                bond_list[i][j] = 1

    for i in range(len(bond_info)):
        if bond_info[i] != '':
            connect_info_i = connect_info[i]
            diff = [i - j for j in connect_info_i]
            min_diff = float('inf')
            for j in range(len(diff)):
                if 0 < diff[j] <= min_diff:
                    min_diff = diff[j]
            x_y_index = diff.index(min_diff)
            a = connect_info_i[x_y_index]
            b = connect_info[a].index(i)
            if bond_info[i] == '-':
                bond_list[i][x_y_index] = 1
                bond_list[a][b] = 1
            elif bond_info[i] == '=':
                bond_list[i][x_y_index] = 2
                bond_list[a][b] = 2
            elif bond_info[i] == '#':
                bond_list[i][x_y_index] = 3
                bond_list[a][b] = 3
            elif bond_info[i] == '∷':
                bond_list[i][x_y_index] = 4
                bond_list[a][b] = 4
    return bond_list


def get_datetime():
    now = datetime.datetime.now()
    now_date = f'{now.year}-{now.month}-{now.day}     {now.hour}:{now.minute}:{now.second}'
    return now_date


def SMILESToMol(SMILES):

    single_element = get_str_list(SMILES=SMILES)
    info_dict = get_info(single_element=single_element)
    connect_info = get_adj(single_element=single_element, info_dict=info_dict)
    bond_list = get_L_bond(info_dict=info_dict, connect_info=connect_info)
    atom_info = info_dict['atom_info']
    atom_info_m = np.matrix(atom_info, dtype=object)
    element_info = atom_info_m[:, 1].flatten().tolist()[0]
    element_index = atom_info_m[:, 0].flatten().tolist()[0]
    ion_valence_state_info = atom_info_m[:, 4].flatten().tolist()[0]
    element_info_1 = [i.upper() if i in aromatic_organic else i for i in element_info]
    ion_valence_state_info_list = []
    ion_valence_state_info_list_1 = []
    for i in range(len(ion_valence_state_info)):
        if ion_valence_state_info[i] == '':
            ion_valence_state_info_list.append('0')
        elif (ion_valence_state_info[i] == '+') | (ion_valence_state_info[i] == '+1'):
            ion_valence_state_info_list.append('3')
            ion_valence_state_info_list_1.append([i + 1, 1])
        elif (ion_valence_state_info[i] == '++') | (ion_valence_state_info[i] == '+2'):
            ion_valence_state_info_list.append('2')
            ion_valence_state_info_list_1.append([i + 1, 2])
        elif (ion_valence_state_info[i] == '+3'):
            ion_valence_state_info_list.append('1')
            ion_valence_state_info_list_1.append([i + 1, 3])
        elif (ion_valence_state_info[i] == '-') | (ion_valence_state_info[i] == '-1'):
            ion_valence_state_info_list.append('5')
            ion_valence_state_info_list_1.append([i + 1, -1])
        elif (ion_valence_state_info[i] == '--') | (ion_valence_state_info[i] == '-2'):
            ion_valence_state_info_list.append('6')
            ion_valence_state_info_list_1.append([i + 1, -2])
        elif (ion_valence_state_info[i] == '-3'):
            ion_valence_state_info_list.append('7')
            ion_valence_state_info_list_1.append([i + 1, -3])
        else:
            ion_valence_state_info_list.append('0')
            print('电荷有问题，请查看')

    Atom_block_list = []
    for i in range(len(element_info_1)):
        x = f'    0.0000    0.0000    0.0000 {element_info_1[i]:<3} 0  {ion_valence_state_info_list[i]:<2} 0  0  0  0  0  0  0  0  0  0\n'
        Atom_block_list.append(x)

    connect_info_f = copy.deepcopy(connect_info)
    bond_list_f = copy.deepcopy(bond_list)
    connect_bond_list = []
    for i in range(len(connect_info_f)):
        if len(connect_info_f[i]) > 0:
            for j in range(len(connect_info_f[i])):
                connect_bond_list.append([i, connect_info_f[i][j], bond_list_f[i][j]])
                i_index = connect_info_f[connect_info_f[i][j]].index(i)
                del connect_info_f[connect_info_f[i][j]][i_index]
                del bond_list_f[connect_info_f[i][j]][i_index]
    connect_bond_list_1 = copy.deepcopy(connect_bond_list)
    for i in range(len(connect_bond_list_1)):
        connect_bond_list_1[i][0] = connect_bond_list_1[i][0] + 1
        connect_bond_list_1[i][1] = connect_bond_list_1[i][1] + 1
        if connect_bond_list_1[i][2] == 1.5:
            connect_bond_list_1[i][2] = 4

    Bond_block_list = []
    for i in range(len(connect_bond_list_1)):
        x = f'{connect_bond_list_1[i][0]:>3}{connect_bond_list_1[i][1]:>3}{connect_bond_list_1[i][2]:>3}  0  0  0  0\n'
        Bond_block_list.append(x)

    Properties_block_list = []
    L = len(ion_valence_state_info_list_1)
    ion_valence_state_info_list_1_f = copy.deepcopy(ion_valence_state_info_list_1)
    while L:
        if L >= 8:
            x = 'M  CHG  8'
            for j in range(8):
                x = x + f'{ion_valence_state_info_list_1_f[0][0]:>4}{ion_valence_state_info_list_1_f[0][1]:>4}'
                del ion_valence_state_info_list_1_f[0]
            Properties_block_list.append(f'{x}\n')
            L -= 8
        else:
            x = f'M  CHG  {L}'
            for j in range(L):
                x = x + f'{ion_valence_state_info_list_1_f[0][0]:>4}{ion_valence_state_info_list_1_f[0][1]:>4}'
                del ion_valence_state_info_list_1_f[0]
            L = 0
            Properties_block_list.append(f'{x}\n')
    Counts_line = f'{len(Atom_block_list):>3}{len(Bond_block_list):>3}  0  0  0  0  0  0  0  0999 V2000\n'
    now_date = get_datetime()
    mol_info_list = ['\n', f'{now_date}\n', '\n',
                     Counts_line] + Atom_block_list + Bond_block_list + Properties_block_list + [
                        'M  END\n', '\n', f'SMILES:{SMILES}']

    return mol_info_list,element_index

