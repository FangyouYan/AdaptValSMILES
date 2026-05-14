import joblib
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

import py_common.MathFunction as mf
import py_common.PositionPropertyDerivative as ppd
from py_common.MathFunction import norm_d, extrapolation_degree
from py_common.StructureInformation import PositionProperty as pp
from py_common.StructureInformation import SIFromMol as SIFX
from py_common.descriptor import MolecularInformation as mi


def hybrid_kernel(X, Y, gamma, lam):
    # linear
    K_linear = linear_kernel(X, Y) / X.shape[1]
    # RBF
    K_rbf = rbf_kernel(X, Y, gamma=gamma)

    return lam * K_linear + (1 - lam) * K_rbf


def svr_hybrid_kernel(X, Y):
    return hybrid_kernel(X, Y, gamma=0.04, lam=0.85)


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


def cal_Tm(ILs_sp, Cation_sp, Anion_sp, Tm_model):
    Pdata_ILs, Sdata_ILs, P_Hdata_ILs, S_Hdata_ILs, IFS_ILs = ILs_sp
    Pdata_Cation, Sdata_Cation, P_Hdata_Cation, S_Hdata_Cation, IFS_Cation = Cation_sp
    Pdata_Anion, Sdata_Anion, P_Hdata_Anion, S_Hdata_Anion, IFS_Anion = Anion_sp

    NI_1 = norm_d(combination_sp(Sdata_ILs['B'], Pdata_ILs['ele'], 1), 6)
    NI_2 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['ion'], 1), 3)
    NI_3 = norm_d(combination_sp(Sdata_ILs['Ccyc_'], Pdata_ILs['wei'], 1), 2)
    NI_4 = norm_d(combination_sp(S_Hdata_ILs['C'], P_Hdata_ILs['noe'], 2), 3)
    NI_5 = norm_d(combination_sp(S_Hdata_ILs['C'], P_Hdata_ILs['bra'], 1), 3)

    NI_6 = norm_d(combination_sp(Sdata_ILs['F'], Pdata_ILs['ion'], 2), 3) / np.sqrt(IFS_ILs[0, 1] / IFS_ILs[0, 3])
    NI_7 = norm_d(combination_sp(S_Hdata_ILs['C'], P_Hdata_ILs['ele'], 2), 3) / np.sqrt(IFS_ILs[0, 1] / IFS_ILs[0, 3])
    NI_8 = norm_d(combination_sp(Sdata_ILs['a'], Pdata_ILs['bra'], 2), 2) / np.sqrt(IFS_ILs[0, 4])
    NI_9 = norm_d(combination_sp(Sdata_ILs['C'], Pdata_ILs['wei'], 2), 3) / np.sqrt(IFS_ILs[0, 4])
    NI_10 = norm_d(combination_sp(Sdata_ILs['C'], Pdata_ILs['noe'], 3), 6) / np.sqrt(IFS_ILs[0, 4])

    NI_11 = norm_d(combination_sp(S_Hdata_ILs['F'], P_Hdata_ILs['bra'], 1), 3) / np.sqrt(IFS_ILs[0, 4])
    NI_12 = norm_d(combination_sp(S_Hdata_ILs['Caro_'], P_Hdata_ILs['ele'], 2), 1) / np.sqrt(IFS_ILs[0, 4])
    NI_13 = norm_d(combination_sp(S_Hdata_ILs['boncyc_'], P_Hdata_ILs['noe'], 2), 3) / np.sqrt(IFS_ILs[0, 4])
    NI_14 = norm_d(combination_sp(Sdata_Cation['C'], Pdata_Cation['noe'], 2), 1)
    NI_15 = norm_d(combination_sp(Sdata_Cation['bon'], Pdata_Cation['ele'], 2), 1)

    NI_16 = norm_d(combination_sp(S_Hdata_Cation['B'], P_Hdata_Cation['ion'], 2), 1)
    NI_17 = norm_d(combination_sp(Sdata_Cation['B'], Pdata_Cation['noe'], 1), 3) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_18 = norm_d(combination_sp(Sdata_Cation['B'], Pdata_Cation['noe'], 2), 1) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_19 = norm_d(combination_sp(Sdata_Cation['C'], Pdata_Cation['ele'], 2), 1) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_20 = norm_d(combination_sp(Sdata_Cation['bon'], Pdata_Cation['rad'], 2), 6) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])

    NI_21 = norm_d(combination_sp(S_Hdata_Cation['bon'], P_Hdata_Cation['bra'], 1), 3) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_22 = norm_d(combination_sp(Sdata_Cation['F'], Pdata_Cation['ele'], 2), 6) / np.sqrt(IFS_Cation[0, 4])

    NI_23 = norm_d(combination_sp(Sdata_Cation['a'], Pdata_Cation['noe_nes'], 4), 6) / np.sqrt(IFS_Cation[0, 4])
    NI_24 = norm_d(combination_sp(S_Hdata_Cation['C'], P_Hdata_Cation['bra'], 2), 3) / np.sqrt(IFS_Cation[0, 4])

    NI_25 = norm_d(combination_sp(S_Hdata_Cation['bon'], P_Hdata_Cation['ele'], 2), 2) / np.sqrt(IFS_Cation[0, 4])
    NI_26 = norm_d(combination_sp(Sdata_Anion['a'], Pdata_Anion['noe'], 2), 3)
    NI_27 = norm_d(combination_sp(Sdata_Anion['C'], Pdata_Anion['noe'], 2), 3)

    NI_28 = norm_d(combination_sp(S_Hdata_Anion['C'], P_Hdata_Anion['bra'], 1), 3)

    NI_29 = norm_d(combination_sp(S_Hdata_Anion['bon'], P_Hdata_Anion['noe'], 2), 4) / np.sqrt(
        IFS_Anion[0, 1] / IFS_Anion[0, 3])
    NI_30 = np.sqrt(norm_d(combination_sp(S_Hdata_Cation['B'], P_Hdata_Cation['bra'], 2), 1) * norm_d(
        combination_sp(S_Hdata_Anion['B'], P_Hdata_Anion['bra'], 2), 1))
    NI_31 = (
                    norm_d(combination_sp(Sdata_Cation['a'], Pdata_Cation['noe'], 3), 6) +
                    norm_d(combination_sp(Sdata_Anion['a'], Pdata_Anion['noe'], 3), 6)
            ) / 2
    NI_32 = (
                    norm_d(combination_sp(Sdata_Cation['bon'], Pdata_Cation['rad'], 1), 3) +
                    norm_d(combination_sp(Sdata_Anion['bon'], Pdata_Anion['rad'], 1), 3)
            ) / 2
    NI_33 = (
                    norm_d(combination_sp(Sdata_Cation['Ccyc_'], Pdata_Cation['noe'], 2), 5) +
                    norm_d(combination_sp(Sdata_Anion['Ccyc_'], Pdata_Anion['noe'], 2), 5)
            ) / 2

    NI_34 = (
                    norm_d(combination_sp(Sdata_Cation['boncyc_'], Pdata_Cation['wei'], 2), 6) +
                    norm_d(combination_sp(Sdata_Anion['boncyc_'], Pdata_Anion['wei'], 2), 6)
            ) / 2
    NI_35 = (
                    norm_d(combination_sp(S_Hdata_Cation['C'], P_Hdata_Cation['rad'], 3), 3) +
                    norm_d(combination_sp(S_Hdata_Anion['C'], P_Hdata_Anion['rad'], 3), 3)
            ) / 2
    NI_36 = (
                    norm_d(combination_sp(S_Hdata_Cation['bon'], P_Hdata_Cation['bra'], 2), 3) +
                    norm_d(combination_sp(S_Hdata_Anion['bon'], P_Hdata_Anion['bra'], 2), 3)
            ) / 2
    NI_37 = (
                    norm_d(combination_sp(S_Hdata_Cation['boncyc_'], P_Hdata_Cation['wei'], 2), 1) +
                    norm_d(combination_sp(S_Hdata_Anion['boncyc_'], P_Hdata_Anion['wei'], 2), 1)
            ) / 2

    X = np.array([[
         NI_1,  NI_2,  NI_3,  NI_4,  NI_5, NI_6,  NI_7,  NI_8,  NI_9,  NI_10,
        NI_11, NI_12, NI_13, NI_14, NI_15, NI_16, NI_17, NI_18, NI_19, NI_20,
        NI_21, NI_22, NI_23, NI_24, NI_25, NI_26, NI_27, NI_28, NI_29, NI_30,
        NI_31, NI_32, NI_33, NI_34, NI_35, NI_36, NI_37,
    ]])

    X_std = np.array([
       5.00525301e+01, 7.14541410e+01, 1.32221111e+04, 4.13721011e+00,
       2.57996637e+01, 7.84900548e+01, 1.60257353e+00, 1.37788093e+00,
       1.24281928e+01, 1.22483704e+02, 8.69303223e-01, 2.30670466e-02,
       5.19365175e-02, 5.06097000e+00, 1.53674837e-01, 3.70409994e+00,
       1.17624533e+01, 2.31308846e+00, 9.80682796e-01, 2.12025848e-01,
       2.67685923e+00, 2.07942216e+00, 8.70682183e-01, 2.61336853e-01,
       2.03916895e-01, 2.81024092e+00, 1.01495571e+01, 2.72431233e+01,
       3.23701921e+00, 3.75383083e+00, 4.26784561e+02, 5.01753501e-01,
       1.11175502e+01, 7.03473433e+00, 1.56219538e+01, 2.13479109e+00,
       9.85059537e-01])

    X_mean = np.array([
       1.86568728e+02, 4.45892779e+02, 1.39448328e+04, 1.72026268e+01,
       7.29901662e+01, 1.03631015e+02, 4.11566357e+00, 4.02929470e+00,
       1.09455787e+01, 5.37681109e+02, 3.68495115e+00, 1.12505978e-02,
       4.32770515e-02, 4.34009541e+01, 6.77559656e-01, 7.21961301e+00,
       7.01987013e+01, 1.34510438e+01, 5.26429115e+00, 5.37856462e-01,
       1.18789794e+01, 4.96841323e+00, 2.62602327e+00, 4.60380279e-01,
       3.28478216e-01, 3.09217243e+00, 9.10569801e+00, 2.11205322e+01,
       3.41983800e+00, 3.89503799e+00, 8.95174072e+02, 1.54963207e+00,
       1.24460133e+01, 5.81746870e+00, 3.35189362e+01, 4.16777396e+00,
       6.05054120e-01])

    X_scaled = (X - X_mean) / X_std

    Tm_cal = Tm_model.predict(X_scaled)[0]

    return Tm_cal


def screen_by_ED(ILs_sp, Cation_sp, Anion_sp, data_NIs):
    Pdata_ILs, Sdata_ILs, P_Hdata_ILs, S_Hdata_ILs, IFS_ILs = ILs_sp
    Pdata_Cation, Sdata_Cation, P_Hdata_Cation, S_Hdata_Cation, IFS_Cation = Cation_sp
    Pdata_Anion, Sdata_Anion, P_Hdata_Anion, S_Hdata_Anion, IFS_Anion = Anion_sp

    NI_1 = norm_d(combination_sp(Sdata_ILs['B'], Pdata_ILs['ele'], 1), 6)
    NI_2 = norm_d(combination_sp(Sdata_ILs['bon'], Pdata_ILs['ion'], 1), 3)
    NI_3 = norm_d(combination_sp(Sdata_ILs['Ccyc_'], Pdata_ILs['wei'], 1), 2)
    NI_4 = norm_d(combination_sp(S_Hdata_ILs['C'], P_Hdata_ILs['noe'], 2), 3)
    NI_5 = norm_d(combination_sp(S_Hdata_ILs['C'], P_Hdata_ILs['bra'], 1), 3)

    NI_6 = norm_d(combination_sp(Sdata_ILs['F'], Pdata_ILs['ion'], 2), 3) / np.sqrt(IFS_ILs[0, 1] / IFS_ILs[0, 3])
    NI_7 = norm_d(combination_sp(S_Hdata_ILs['C'], P_Hdata_ILs['ele'], 2), 3) / np.sqrt(IFS_ILs[0, 1] / IFS_ILs[0, 3])
    NI_8 = norm_d(combination_sp(Sdata_ILs['a'], Pdata_ILs['bra'], 2), 2) / np.sqrt(IFS_ILs[0, 4])
    NI_9 = norm_d(combination_sp(Sdata_ILs['C'], Pdata_ILs['wei'], 2), 3) / np.sqrt(IFS_ILs[0, 4])
    NI_10 = norm_d(combination_sp(Sdata_ILs['C'], Pdata_ILs['noe'], 3), 6) / np.sqrt(IFS_ILs[0, 4])

    NI_11 = norm_d(combination_sp(S_Hdata_ILs['F'], P_Hdata_ILs['bra'], 1), 3) / np.sqrt(IFS_ILs[0, 4])
    NI_12 = norm_d(combination_sp(S_Hdata_ILs['Caro_'], P_Hdata_ILs['ele'], 2), 1) / np.sqrt(IFS_ILs[0, 4])
    NI_13 = norm_d(combination_sp(S_Hdata_ILs['boncyc_'], P_Hdata_ILs['noe'], 2), 3) / np.sqrt(IFS_ILs[0, 4])
    NI_14 = norm_d(combination_sp(Sdata_Cation['C'], Pdata_Cation['noe'], 2), 1)
    NI_15 = norm_d(combination_sp(Sdata_Cation['bon'], Pdata_Cation['ele'], 2), 1)

    NI_16 = norm_d(combination_sp(S_Hdata_Cation['B'], P_Hdata_Cation['ion'], 2), 1)
    NI_17 = norm_d(combination_sp(Sdata_Cation['B'], Pdata_Cation['noe'], 1), 3) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_18 = norm_d(combination_sp(Sdata_Cation['B'], Pdata_Cation['noe'], 2), 1) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_19 = norm_d(combination_sp(Sdata_Cation['C'], Pdata_Cation['ele'], 2), 1) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_20 = norm_d(combination_sp(Sdata_Cation['bon'], Pdata_Cation['rad'], 2), 6) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])

    NI_21 = norm_d(combination_sp(S_Hdata_Cation['bon'], P_Hdata_Cation['bra'], 1), 3) / np.sqrt(
        IFS_Cation[0, 1] / IFS_Cation[0, 3])
    NI_22 = norm_d(combination_sp(Sdata_Cation['F'], Pdata_Cation['ele'], 2), 6) / np.sqrt(IFS_Cation[0, 4])

    NI_23 = norm_d(combination_sp(Sdata_Cation['a'], Pdata_Cation['noe_nes'], 4), 6) / np.sqrt(IFS_Cation[0, 4])
    NI_24 = norm_d(combination_sp(S_Hdata_Cation['C'], P_Hdata_Cation['bra'], 2), 3) / np.sqrt(IFS_Cation[0, 4])

    NI_25 = norm_d(combination_sp(S_Hdata_Cation['bon'], P_Hdata_Cation['ele'], 2), 2) / np.sqrt(IFS_Cation[0, 4])
    NI_26 = norm_d(combination_sp(Sdata_Anion['a'], Pdata_Anion['noe'], 2), 3)
    NI_27 = norm_d(combination_sp(Sdata_Anion['C'], Pdata_Anion['noe'], 2), 3)

    NI_28 = norm_d(combination_sp(S_Hdata_Anion['C'], P_Hdata_Anion['bra'], 1), 3)

    NI_29 = norm_d(combination_sp(S_Hdata_Anion['bon'], P_Hdata_Anion['noe'], 2), 4) / np.sqrt(
        IFS_Anion[0, 1] / IFS_Anion[0, 3])
    NI_30 = np.sqrt(norm_d(combination_sp(S_Hdata_Cation['B'], P_Hdata_Cation['bra'], 2), 1) * norm_d(
        combination_sp(S_Hdata_Anion['B'], P_Hdata_Anion['bra'], 2), 1))
    NI_31 = (
                    norm_d(combination_sp(Sdata_Cation['a'], Pdata_Cation['noe'], 3), 6) +
                    norm_d(combination_sp(Sdata_Anion['a'], Pdata_Anion['noe'], 3), 6)
            ) / 2
    NI_32 = (
                    norm_d(combination_sp(Sdata_Cation['bon'], Pdata_Cation['rad'], 1), 3) +
                    norm_d(combination_sp(Sdata_Anion['bon'], Pdata_Anion['rad'], 1), 3)
            ) / 2
    NI_33 = (
                    norm_d(combination_sp(Sdata_Cation['Ccyc_'], Pdata_Cation['noe'], 2), 5) +
                    norm_d(combination_sp(Sdata_Anion['Ccyc_'], Pdata_Anion['noe'], 2), 5)
            ) / 2

    NI_34 = (
                    norm_d(combination_sp(Sdata_Cation['boncyc_'], Pdata_Cation['wei'], 2), 6) +
                    norm_d(combination_sp(Sdata_Anion['boncyc_'], Pdata_Anion['wei'], 2), 6)
            ) / 2
    NI_35 = (
                    norm_d(combination_sp(S_Hdata_Cation['C'], P_Hdata_Cation['rad'], 3), 3) +
                    norm_d(combination_sp(S_Hdata_Anion['C'], P_Hdata_Anion['rad'], 3), 3)
            ) / 2
    NI_36 = (
                    norm_d(combination_sp(S_Hdata_Cation['bon'], P_Hdata_Cation['bra'], 2), 3) +
                    norm_d(combination_sp(S_Hdata_Anion['bon'], P_Hdata_Anion['bra'], 2), 3)
            ) / 2
    NI_37 = (
                    norm_d(combination_sp(S_Hdata_Cation['boncyc_'], P_Hdata_Cation['wei'], 2), 1) +
                    norm_d(combination_sp(S_Hdata_Anion['boncyc_'], P_Hdata_Anion['wei'], 2), 1)
            ) / 2

    X = np.array([[
         NI_1,  NI_2,  NI_3,  NI_4,  NI_5, NI_6,  NI_7,  NI_8,  NI_9,  NI_10,
        NI_11, NI_12, NI_13, NI_14, NI_15, NI_16, NI_17, NI_18, NI_19, NI_20,
        NI_21, NI_22, NI_23, NI_24, NI_25, NI_26, NI_27, NI_28, NI_29, NI_30,
        NI_31, NI_32, NI_33, NI_34, NI_35, NI_36, NI_37,
    ]])

    X_std = np.array([
       5.00525301e+01, 7.14541410e+01, 1.32221111e+04, 4.13721011e+00,
       2.57996637e+01, 7.84900548e+01, 1.60257353e+00, 1.37788093e+00,
       1.24281928e+01, 1.22483704e+02, 8.69303223e-01, 2.30670466e-02,
       5.19365175e-02, 5.06097000e+00, 1.53674837e-01, 3.70409994e+00,
       1.17624533e+01, 2.31308846e+00, 9.80682796e-01, 2.12025848e-01,
       2.67685923e+00, 2.07942216e+00, 8.70682183e-01, 2.61336853e-01,
       2.03916895e-01, 2.81024092e+00, 1.01495571e+01, 2.72431233e+01,
       3.23701921e+00, 3.75383083e+00, 4.26784561e+02, 5.01753501e-01,
       1.11175502e+01, 7.03473433e+00, 1.56219538e+01, 2.13479109e+00,
       9.85059537e-01])

    X_mean = np.array([
       1.86568728e+02, 4.45892779e+02, 1.39448328e+04, 1.72026268e+01,
       7.29901662e+01, 1.03631015e+02, 4.11566357e+00, 4.02929470e+00,
       1.09455787e+01, 5.37681109e+02, 3.68495115e+00, 1.12505978e-02,
       4.32770515e-02, 4.34009541e+01, 6.77559656e-01, 7.21961301e+00,
       7.01987013e+01, 1.34510438e+01, 5.26429115e+00, 5.37856462e-01,
       1.18789794e+01, 4.96841323e+00, 2.62602327e+00, 4.60380279e-01,
       3.28478216e-01, 3.09217243e+00, 9.10569801e+00, 2.11205322e+01,
       3.41983800e+00, 3.89503799e+00, 8.95174072e+02, 1.54963207e+00,
       1.24460133e+01, 5.81746870e+00, 3.35189362e+01, 4.16777396e+00,
       6.05054120e-01])


    X_scaled = (X - X_mean) / X_std

    # (back,forw)
    ED_domain = np.array([
         [0.084, 0.110], [0.000, 0.194], [0.000, 0.127], [0.067, 0.041], [0.064, 0.058],
         [0.070, 0.000], [0.194, 0.052], [0.065, 0.144], [0.065, 0.124], [0.095, 0.109],
         [0.109, 0.135], [0.000, 0.103], [0.000, 0.137], [0.247, 0.341], [0.198, 0.385],
         [0.268, 0.320], [0.259, 0.331], [0.256, 0.308], [0.284, 0.318], [0.211, 0.394],
         [0.310, 0.269], [0.243, 0.336], [0.181, 0.403], [0.187, 0.395], [0.210, 0.385],
         [0.000, 0.171], [0.000, 0.111], [0.000, 0.000], [0.000, 0.152], [0.000, 0.121],
         [0.062, 0.148], [0.030, 0.167], [0.000, 0.142], [0.000, 0.121], [0.073, 0.063],
         [0.000, 0.126], [0.000, 0.220]])
    new_ED_domain_list = np.array([[f'[0, {row[0]}]', f'[0, {row[1]}]'] for row in ED_domain])

    extrap_directions = []
    ED_values = []
    for se_i in range(37):
        ED_value_i, extrap_direction_i = extrapolation_degree(x_train=data_NIs, x_test=X_scaled, se_i=se_i,
                                                              main_weight=0.5)
        extrap_directions.append(extrap_direction_i)
        ED_values.append(ED_value_i)

    extrap_directions_a = np.array(extrap_directions)
    ED_values_a = np.array(ED_values)

    conditions = [
        extrap_directions_a == 'backward',
        extrap_directions_a == 'forward'
    ]

    choices = [
        ED_values_a - ED_domain[:, 0],
        ED_values_a - ED_domain[:, 1]
    ]

    ED_diff = np.select(conditions, choices, default=0.0)

    if np.any(ED_diff > 0):
        return 'Reject'
    else:
        return 'Accept'






if __name__ == '__main__':
    cation_name = '[C4mim]'
    anion_name = '[PF6]'
    ILs_name = f'{cation_name}{anion_name}'
    cation_s=rf'./test_structure/{cation_name}.mol'
    anion_s = rf'./test_structure/{anion_name}.mol'
    ILs_s = rf'./test_structure/{ILs_name}.mol'

    Tm_model = joblib.load(f'Tm_svr_model.pkl')

    with open(ILs_s, 'r') as f:
        ILs_ISI = f.readlines()
    ILs_sp = calculate_SP(ILs_ISI)

    with open(cation_s, 'r') as f:
        cation_ISI = f.readlines()
    cation_sp = calculate_SP(cation_ISI)

    with open(anion_s, 'r') as f:
        anion_ISI = f.readlines()
    anion_sp = calculate_SP(anion_ISI)

    Tm_cal = cal_Tm(ILs_sp=ILs_sp, Cation_sp=cation_sp, Anion_sp=anion_sp,Tm_model=Tm_model)
    print(f'{ILs_name}: {Tm_cal} K')