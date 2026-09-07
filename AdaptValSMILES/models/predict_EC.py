import numpy as np

import py_common.MathFunction as mf
import py_common.PositionPropertyDerivative as ppd
from py_common.MathFunction import norm_d, extrapolation_degree
from py_common.StructureInformation import PositionProperty as pp
from py_common.StructureInformation import SIFromMol as SIFX
from py_common.descriptor import MolecularInformation as mi


def calculate_SP(ISIX, step_max=None):
    SI_X = SIFX(ISI=ISIX, step_max=step_max)
    Latom = SI_X.Latom
    P = pp.properties(name_atom=Latom)
    S = pp.step(SI=SI_X)

    iH = SI_X.iH

    P_H = mf.DicDelete(iD=iH, SI=P)
    S_H = mf.DicDelete(iD=iH, SI=S)

    P['bra'] = sum(S['a']).reshape(-1, 1)
    P_H['bra'] = sum(S_H['a']).reshape(-1, 1)

    Pdata, Ptype = ppd.properties(P=P, label='P')
    Sdata, Stype = ppd.step(S=S, label='S')

    P_Hdata, P_Htype = ppd.properties(P=P_H, label='P_deH')
    S_Hdata, S_Htype = ppd.step(S=S_H, label='S_deH')

    calc_keys = ['n_atom', 'n_nonH', 'mw', 'st_m', 'st_s']
    IFS = mi.InformationFromStep(Latom=Latom, Mwei=P['wei'], SF_H=S_H['F'], calc_keys=calc_keys)

    X_sp = (Pdata, Sdata, P_Hdata, S_Hdata, IFS)

    return X_sp


def combination_sp(S, P, combination_i):
    if combination_i == 1:
        return np.multiply(S, (P.dot(P.T)))
    elif combination_i == 2:
        return np.multiply(S, abs(P - P.T))
    elif combination_i == 3:
        return S.dot(P.dot(P.T))
    elif combination_i == 4:
        return S.dot(abs(P - P.T))
    return ValueError("combination_i must be 1–4")


def cal_EC(ILs_sp, Cation_sp, Anion_sp, T):
    Pdata_ILs, Sdata_ILs, P_Hdata_ILs, S_Hdata_ILs, IFS_ILs = ILs_sp
    Pdata_Cation, Sdata_Cation, P_Hdata_Cation, S_Hdata_Cation, IFS_Cation = Cation_sp
    Pdata_Anion, Sdata_Anion, P_Hdata_Anion, S_Hdata_Anion, IFS_Anion = Anion_sp

    NI_1 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['ion'], 1), 3)
    NI_2 = norm_d(combination_sp(S_Hdata_ILs['boncyc_'], P_Hdata_ILs['noe'], 4), 6)
    NI_3 = norm_d(combination_sp(Sdata_ILs['C'], Pdata_ILs['ele'], 1), 4) / np.sqrt(IFS_ILs[0, 4])
    NI_4 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['ele'], 1), 3) / np.sqrt(IFS_ILs[0, 4])
    NI_5 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['rad'], 2), 3) / np.sqrt(IFS_ILs[0, 4])

    NI_6 = norm_d(combination_sp(S_Hdata_Cation['B'], P_Hdata_Cation['rad'], 1), 3)
    NI_7 = norm_d(combination_sp(S_Hdata_Cation['C'], P_Hdata_Cation['bra'], 1), 1)
    NI_8 = norm_d(combination_sp(Sdata_Cation['a'], Pdata_Cation['ion'], 2), 6) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_9 = norm_d(combination_sp(Sdata_Cation['B'], Pdata_Cation['noe_nes'], 1), 3) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_10 = norm_d(combination_sp(Sdata_Cation['bon'], Pdata_Cation['ion'], 2), 6) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])

    NI_11 = norm_d(combination_sp(S_Hdata_Cation['a'], P_Hdata_Cation['noe'], 4), 6) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_12 = norm_d(combination_sp(S_Hdata_Cation['B'], P_Hdata_Cation['ele'], 1), 3) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_13 = norm_d(combination_sp(Sdata_Cation['F'], Pdata_Cation['bra'], 2), 3) / np.sqrt(IFS_Cation[0, 4])
    NI_14 = norm_d(combination_sp(Sdata_ILs['B'], Pdata_ILs['rad'], 1), 5) / T
    NI_15 = norm_d(combination_sp(Sdata_ILs['a'], Pdata_ILs['rad'], 2), 1) / np.sqrt(IFS_ILs[0, 1] / IFS_ILs[0, 3]) / T

    NI_16 = norm_d(combination_sp(Sdata_ILs['C'], Pdata_ILs['ele'], 1), 3) / np.sqrt(IFS_ILs[0, 4]) / T
    NI_17 = norm_d(combination_sp(Sdata_ILs['C'], Pdata_ILs['rad'], 4), 6) / np.sqrt(IFS_ILs[0, 4]) / T
    NI_18 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['ele'], 2), 2) / np.sqrt(IFS_ILs[0, 4]) / T
    NI_19 = norm_d(combination_sp(Sdata_Cation['B'], Pdata_Cation['wei'], 1), 1) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3]) / T
    NI_20 = norm_d(combination_sp(Sdata_Cation['C'], Pdata_Cation['bra'], 1), 3) / np.sqrt(IFS_Cation[0, 4]) / T

    NI_21 = norm_d(combination_sp(S_Hdata_Cation['C'], P_Hdata_Cation['ion'], 4), 3) / np.sqrt(IFS_Cation[0, 4]) / T
    NI_22 = norm_d(combination_sp(Sdata_Anion['a'], Pdata_Anion['rad'], 1), 6) / np.sqrt(
        IFS_Anion[0, 1] / IFS_Anion[0, 3]) / T
    NI_23 = norm_d(combination_sp(Sdata_Anion['B'], Pdata_Anion['noe'], 1), 3) / np.sqrt(
        IFS_Anion[0, 1] / IFS_Anion[0, 3]) / T
    NI_24 = norm_d(combination_sp(Sdata_ILs['a'], Pdata_ILs['noe'], 3), 3) / np.sqrt(
        IFS_ILs[0, 1] / IFS_ILs[0, 3]) / T ** 2
    NI_25 = norm_d(combination_sp(S_Hdata_ILs['bon'], P_Hdata_ILs['rad'], 3), 3) / np.sqrt(
        IFS_ILs[0, 1] / IFS_ILs[0, 3]) / T ** 2

    NI_26 = norm_d(combination_sp(S_Hdata_ILs['bon'], P_Hdata_ILs['bra'], 1), 1) / np.sqrt(
        IFS_ILs[0, 1] / IFS_ILs[0, 3]) / T ** 2
    NI_27 = norm_d(combination_sp(S_Hdata_ILs['bon'], P_Hdata_ILs['bra'], 2), 3) / np.sqrt(
        IFS_ILs[0, 1] / IFS_ILs[0, 3]) / T ** 2
    NI_28 = norm_d(combination_sp(Sdata_ILs['a'], Pdata_ILs['noe'], 4), 4) / np.sqrt(IFS_ILs[0, 4]) / T ** 2
    NI_29 = norm_d(combination_sp(Sdata_ILs['B'], Pdata_ILs['rad'], 2), 6) / np.sqrt(IFS_ILs[0, 4]) / T ** 2
    NI_30 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['rad'], 1), 3) / np.sqrt(IFS_ILs[0, 4]) / T ** 2

    NI_31 = norm_d(combination_sp(S_Hdata_Cation['bon'], P_Hdata_Cation['rad'], 1), 6) / T ** 2
    NI_32 = norm_d(combination_sp(S_Hdata_Cation['boncyc_'], P_Hdata_Cation['bra'], 2), 2) / T ** 2
    NI_33 = norm_d(combination_sp(S_Hdata_Cation['B'], P_Hdata_Cation['noe'], 2), 6) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3]) / T ** 2
    NI_34 = norm_d(combination_sp(Sdata_Anion['bon'], Pdata_Anion['noe'], 2), 4) / np.sqrt(
        IFS_Anion[0, 1] / IFS_Anion[0, 3]) / T ** 2

    X = np.array([[
        1,
        NI_1, NI_2, NI_3, NI_4, NI_5, NI_6, NI_7, NI_8, NI_9, NI_10,
        NI_11, NI_12, NI_13, NI_14, NI_15, NI_16, NI_17, NI_18, NI_19, NI_20,
        NI_21, NI_22, NI_23, NI_24, NI_25, NI_26, NI_27, NI_28, NI_29, NI_30,
        NI_31, NI_32, NI_33, NI_34
    ]])

    bb = np.array([
        -7.99486047e+00,
        -8.02443977e-03, -1.40954496e-02, 1.73768795e-01, 1.86738238e+00, -3.71918869e+01,
        9.45294366e-01, 2.20767064e-02, -9.63804187e-01, 4.84870712e-01, 7.70255669e-01,
        -2.61081626e-02, -3.09745830e-01, 1.04844485e+00, 2.68831627e+02, 1.38992032e+04,
        3.25562388e+02, -7.64553493e+01, -1.40844011e+03, -2.87932061e+00, -5.86578684e+01,
        6.18870930e+01, -1.38012017e+02, 1.11035461e+01, -2.83151555e+03, 4.52838405e+04,
        -3.34156227e+04, -1.78768944e+04, -3.99584596e+04, 3.18696347e+05, -1.24091241e+06,
        -8.52468784e+04, 4.39008828e+03, 1.80492192e+04, -1.01832228e+04
    ])

    EC_cal = np.sum(X * bb)

    return EC_cal


def screen_by_ED(ILs_sp, Cation_sp, Anion_sp,data_NIs):
    Pdata_ILs, Sdata_ILs, P_Hdata_ILs, S_Hdata_ILs, IFS_ILs = ILs_sp
    Pdata_Cation, Sdata_Cation, P_Hdata_Cation, S_Hdata_Cation, IFS_Cation = Cation_sp
    Pdata_Anion, Sdata_Anion, P_Hdata_Anion, S_Hdata_Anion, IFS_Anion = Anion_sp

    NI_1 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['ion'], 1), 3)
    NI_2 = norm_d(combination_sp(S_Hdata_ILs['boncyc_'], P_Hdata_ILs['noe'], 4), 6)
    NI_3 = norm_d(combination_sp(Sdata_ILs['C'], Pdata_ILs['ele'], 1), 4) / np.sqrt(IFS_ILs[0, 4])
    NI_4 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['ele'], 1), 3) / np.sqrt(IFS_ILs[0, 4])
    NI_5 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['rad'], 2), 3) / np.sqrt(IFS_ILs[0, 4])

    NI_6 = norm_d(combination_sp(S_Hdata_Cation['B'], P_Hdata_Cation['rad'], 1), 3)
    NI_7 = norm_d(combination_sp(S_Hdata_Cation['C'], P_Hdata_Cation['bra'], 1), 1)
    NI_8 = norm_d(combination_sp(Sdata_Cation['a'], Pdata_Cation['ion'], 2), 6) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_9 = norm_d(combination_sp(Sdata_Cation['B'], Pdata_Cation['noe_nes'], 1), 3) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_10 = norm_d(combination_sp(Sdata_Cation['bon'], Pdata_Cation['ion'], 2), 6) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])

    NI_11 = norm_d(combination_sp(S_Hdata_Cation['a'], P_Hdata_Cation['noe'], 4), 6) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_12 = norm_d(combination_sp(S_Hdata_Cation['B'], P_Hdata_Cation['ele'], 1), 3) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_13 = norm_d(combination_sp(Sdata_Cation['F'], Pdata_Cation['bra'], 2), 3) / np.sqrt(IFS_Cation[0, 4])
    NI_14 = norm_d(combination_sp(Sdata_ILs['B'], Pdata_ILs['rad'], 1), 5)
    NI_15 = norm_d(combination_sp(Sdata_ILs['a'], Pdata_ILs['rad'], 2), 1) / np.sqrt(IFS_ILs[0, 1] / IFS_ILs[0, 3])

    NI_16 = norm_d(combination_sp(Sdata_ILs['C'], Pdata_ILs['ele'], 1), 3) / np.sqrt(IFS_ILs[0, 4])
    NI_17 = norm_d(combination_sp(Sdata_ILs['C'], Pdata_ILs['rad'], 4), 6) / np.sqrt(IFS_ILs[0, 4])
    NI_18 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['ele'], 2), 2) / np.sqrt(IFS_ILs[0, 4])
    NI_19 = norm_d(combination_sp(Sdata_Cation['B'], Pdata_Cation['wei'], 1), 1) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_20 = norm_d(combination_sp(Sdata_Cation['C'], Pdata_Cation['bra'], 1), 3) / np.sqrt(IFS_Cation[0, 4])

    NI_21 = norm_d(combination_sp(S_Hdata_Cation['C'], P_Hdata_Cation['ion'], 4), 3) / np.sqrt(IFS_Cation[0, 4])
    NI_22 = norm_d(combination_sp(Sdata_Anion['a'], Pdata_Anion['rad'], 1), 6) / np.sqrt(
        IFS_Anion[0, 1] / IFS_Anion[0, 3])
    NI_23 = norm_d(combination_sp(Sdata_Anion['B'], Pdata_Anion['noe'], 1), 3) / np.sqrt(
        IFS_Anion[0, 1] / IFS_Anion[0, 3])
    NI_24 = norm_d(combination_sp(Sdata_ILs['a'], Pdata_ILs['noe'], 3), 3) / np.sqrt(IFS_ILs[0, 1] / IFS_ILs[0, 3])
    NI_25 = norm_d(combination_sp(S_Hdata_ILs['bon'], P_Hdata_ILs['rad'], 3), 3) / np.sqrt(
        IFS_ILs[0, 1] / IFS_ILs[0, 3])

    NI_26 = norm_d(combination_sp(S_Hdata_ILs['bon'], P_Hdata_ILs['bra'], 1), 1) / np.sqrt(
        IFS_ILs[0, 1] / IFS_ILs[0, 3])
    NI_27 = norm_d(combination_sp(S_Hdata_ILs['bon'], P_Hdata_ILs['bra'], 2), 3) / np.sqrt(
        IFS_ILs[0, 1] / IFS_ILs[0, 3])
    NI_28 = norm_d(combination_sp(Sdata_ILs['a'], Pdata_ILs['noe'], 4), 4) / np.sqrt(IFS_ILs[0, 4])
    NI_29 = norm_d(combination_sp(Sdata_ILs['B'], Pdata_ILs['rad'], 2), 6) / np.sqrt(IFS_ILs[0, 4])
    NI_30 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['rad'], 1), 3) / np.sqrt(IFS_ILs[0, 4])

    NI_31 = norm_d(combination_sp(S_Hdata_Cation['bon'], P_Hdata_Cation['rad'], 1), 6)
    NI_32 = norm_d(combination_sp(S_Hdata_Cation['boncyc_'], P_Hdata_Cation['bra'], 2), 2)
    NI_33 = norm_d(combination_sp(S_Hdata_Cation['B'], P_Hdata_Cation['noe'], 2), 6) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_34 = norm_d(combination_sp(Sdata_Anion['bon'], Pdata_Anion['noe'], 2), 4) / np.sqrt(
        IFS_Anion[0, 1] / IFS_Anion[0, 3])

    NIs = np.array([[
        NI_1, NI_2, NI_3, NI_4, NI_5, NI_6, NI_7, NI_8, NI_9, NI_10,
        NI_11, NI_12, NI_13, NI_14, NI_15, NI_16, NI_17, NI_18, NI_19, NI_20,
        NI_21, NI_22, NI_23, NI_24, NI_25, NI_26, NI_27, NI_28, NI_29, NI_30,
        NI_31, NI_32, NI_33, NI_34
    ]])

    # (back,forw)
    ED_domain = np.array([
        [0.094, 0.273], [0.000, 0.285], [0.158, 0.328], [0.135, 0.278], [0.121, 0.241],
        [0.420, 0.473], [0.446, 0.273], [0.376, 0.330], [0.411, 0.245], [0.480, 0.203],
        [0.259, 0.399], [0.378, 0.283], [0.313, 0.313], [0.107, 0.164], [0.099, 0.167],
        [0.125, 0.276], [0.000, 0.280], [0.072, 0.208], [0.165, 0.180], [0.223, 0.295],
        [0.261, 0.182], [0.000, 0.260], [0.263, 0.000], [0.128, 0.081], [0.090, 0.196],
        [0.113, 0.114], [0.097, 0.230], [0.000, 0.168], [0.029, 0.152], [0.073, 0.246],
        [0.123, 0.099], [0.000, 0.304], [0.145, 0.217], [0.000, 0.110]])
    # new_ED_domain_list = np.array([[f'[0, {row[0]}]', f'[0, {row[1]}]'] for row in ED_domain])
    extrap_directions = []
    ED_values= []
    for se_i in range(34):
        ED_value_i,extrap_direction_i = extrapolation_degree(x_train=data_NIs, x_test=NIs, se_i=se_i, main_weight=0.5)
        extrap_directions.append(extrap_direction_i)
        ED_values.append(ED_value_i)

    extrap_directions_a=np.array(extrap_directions)
    ED_values_a=np.array(ED_values)

    conditions = [
        extrap_directions_a == 'backward',
        extrap_directions_a == 'forward'
    ]

    choices = [
        ED_values_a - ED_domain[:, 0],
        ED_values_a - ED_domain[:, 1]
    ]

    ED_diff = np.select(conditions, choices, default=0.0)

    if np.any(ED_diff>0):
        return 'Reject'
    else:
        return 'Accept'



if __name__ == '__main__':
    cation_name = '[C4mim]'
    anion_name = '[PF6]'
    Tem = 298.15  # K
    ILs_name = f'{cation_name}{anion_name}'
    cation_s=rf'./test_structure/{cation_name}.mol'
    anion_s = rf'./test_structure/{anion_name}.mol'
    ILs_s = rf'./test_structure/{ILs_name}.mol'

    with open(ILs_s, 'r') as f:
        ILs_ISI = f.readlines()
    ILs_sp = calculate_SP(ILs_ISI)

    with open(cation_s, 'r') as f:
        cation_ISI = f.readlines()
    cation_sp = calculate_SP(cation_ISI)

    with open(anion_s, 'r') as f:
        anion_ISI = f.readlines()
    anion_sp = calculate_SP(anion_ISI)

    EC_cal = cal_EC(ILs_sp=ILs_sp, Cation_sp=cation_sp, Anion_sp=anion_sp,T=Tem)
    print(f'{ILs_name}: {EC_cal}')