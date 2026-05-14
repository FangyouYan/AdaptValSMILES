import copy

import numpy as np

from .AtomicProperty import AtomicProperty

class CommonUtils:
    @staticmethod
    def AdjacentLists(Latom, n_atom, Sa, Sbon):
        """
        get the adjacent relationship of atoms
        :param Latom: atom type,list
        :param n_atom: the number of atoms,int
        :param Sa: adjacency matrix, array
        :param Sbon: bond  matrix, array
        :return:  Ladj(adjacent relationship of atoms,list,start 1),
                 Ladj_H(adjacent relationship of hydrogen atoms with all atoms,list,start 1),
                 Ladj_nH(adjacent relationship of none hydrogen atoms with all atoms,list,start 1),
                 Lbon(the chemical bond,list)
        """
        Ladj_H = []
        Ladj_nH = []
        Ladj = []
        Lbon = []
        for j in range(0, n_atom):
            Ladj_ = np.where(Sa[j, :] > 0)
            Ladj.append(list(1 + Ladj_[0]))
            Lbon.append(Sbon[j, Ladj_].tolist()[0])
            Ladj__ = []
            Ladj_nH_ = []
            for i in Ladj_[0]:
                if Latom[i] == 'H':
                    Ladj__.append(i + 1)
                else:
                    Ladj_nH_.append(i + 1)
            Ladj_nH.append(Ladj_nH_)
            Ladj_H.append(Ladj__)
        return Ladj, Ladj_H, Ladj_nH, Lbon


    @staticmethod
    def FullStep(n_atom, Sa, Ladj,step_max=None):
        """
        Get full step matrix (SF)
        :param n_atom: the number of atoms
        :param Sa: adjacency matrix
        :param Ladj: adjacent relationship of atoms,list
        :return: SF
        """
        if step_max==None:
            step_max = n_atom-1
        SF = copy.deepcopy(Sa)
        w_ms_r, w_ms_c = np.where(SF == 1)
        for m in range(1, step_max):
            if len(w_ms_r) == 0:
                break
            w_ms_r_ = []
            w_ms_c_ = []
            for i in range(0, len(w_ms_r)):
                w_msi = Ladj[w_ms_c[i]]
                for j in w_msi:
                    if SF[w_ms_r[i], j - 1] == 0 and w_ms_r[i] != j - 1:
                        SF[w_ms_r[i], j - 1] = m + 1
                        w_ms_r_.append(w_ms_r[i])
                        w_ms_c_.append(j - 1)
            w_ms_r = w_ms_r_
            w_ms_c = w_ms_c_
        return SF

    @staticmethod
    def Aromatics(n_atom, Lbon, b_aro):
        """
        identify aromatic atoms
        :param n_atom: the number of atoms
        :param Lbon: the chemical bond
        :param b_aro: the bond in aromatics ring
        :return:Paro(the aromatics atoms, array)
        """
        Paro = np.zeros((n_atom, 1))
        for k in range(n_atom):
            if len(np.where(np.array(Lbon[k]) == b_aro)[0]) > 0:
                Paro[k, 0] = 1
        return Paro

    @staticmethod
    def DeleteEndAtom(Ladj_nH, Latom):
        """
        delete the atoms in the end
        :param Ladj_nH: adjacent relationship of none hydrogen atoms with all atoms
        :param Latom: atom type
        :return: Ladj_nl(adjacent relationship of  atoms not in the end,start 1),
                 lne(the index of atom not in the end, start 0)
        """
        Ladj_nl = copy.deepcopy(Ladj_nH)
        lne = []
        for i, j in enumerate(Ladj_nl):
            if Latom[i] == 'H':
                Ladj_nl[i] = []
            elif len(j) > 0:
                lne.append(i)
        isem = 1
        while isem > 0:
            isem = 0
            for i in lne:
                if len(Ladj_nl[i]) == 1:
                    isem = 1
                    ii_0 = Ladj_nl[i][0] - 1
                    ii_1 = i + 1
                    Ladj_nl[ii_0].remove(ii_1)
                    Ladj_nl[i] = []
                    lne.remove(i)
        return Ladj_nl, lne

    @staticmethod
    def AtomInRing(SF, Ladj_nl, lne, Paro):
        """
        indentify atoms in the ring
        :param SF: full step matrix,array
        :param Ladj_nl:adjacent relationship of  atoms not in the end,start 1
        :param lne:the index of atom not in the end, start 0
        :param Paro:the aromatics atoms, array
        :return:Cyc(atoms in the ring including aromatic atoms, array)
        """
        Cyc = copy.deepcopy(Paro)
        while [] in Ladj_nl:
            Ladj_nl.remove([])
        Ladj_nl_ = copy.deepcopy(Ladj_nl)
        lne_ = copy.deepcopy(lne)
        w_a = np.where(Paro == 1)[0]
        for j in w_a:
            i_j = lne_.index(j)
            lne_.remove(j)
            del Ladj_nl_[i_j]
        for i, k in enumerate(Ladj_nl_):
            i_l = lne_[i]
            for j in lne:
                wl = (np.where(SF[j, (np.array(k) - 1).tolist()] < SF[j, i_l] + 1)[0]).tolist()
                if len(wl) > 1:
                    Cyc[i_l, 0] = 1
                    break
        return Cyc

    @staticmethod
    def BondNumber2Line(Lbon):
        """
        Change the bond from number to liner str
        :param Lbon: the bond in number
        :return: Lbon_str(the bond in liner str)
        """
        Lbon_str = []
        num2lin = {1: '-', 2: '=', 3: '≡', 1.5: '∷'}
        for i, j in enumerate(Lbon):
            bon_str = []
            for k in j:
                bon_str.append(num2lin[k])
            Lbon_str.append(bon_str)
        return Lbon_str

    @staticmethod
    def BondLine2Curve(Lbon_str, SF, Ladj, Ladj_nl, lne):
        """
        Change the liner str to noneline str in the ring
        :param Lbon_str: the bond in liner str
        :param SF: full step matrix
        :param Ladj: adjacent relationship of atoms
        :param Ladj_nl: adjacent relationship of atoms not in the end
        :param lne: the index of atom not in the end
        :return: Lbon_str(the bond in noliner str)
        """
        lin2nlin = {'-': '～', '=': '≈', '≡': '≋', '∷': '∷'}
        lin2nlin_keys = lin2nlin.keys()
        while [] in Ladj_nl:
            Ladj_nl.remove([])
        for i, k in enumerate(Ladj_nl):
            i_l = lne[i]
            for j in lne:
                wl = (np.where(SF[j, (np.array(k) - 1)] < SF[j, i_l] + 1)[0]).tolist()
                if len(wl) > 1:
                    for k_wl in wl:
                        i_s = Ladj[i_l].index(k[k_wl])
                        if Lbon_str[i_l][i_s] in lin2nlin_keys:
                            Lbon_str[i_l][i_s] = lin2nlin[Lbon_str[i_l][i_s]]
        return Lbon_str


    @staticmethod
    def FullDistance(Mdis):
        """
        The calculation of full distance matrix (DF)
        :param Mdis:atomic coordinates
        :return:DF(full distance matrix,array)
        """
        Mdis = copy.deepcopy(Mdis)
        diff = Mdis[:, np.newaxis, :] - Mdis[np.newaxis, :, :]  # 广播机制
        DF = np.sqrt(np.sum(diff ** 2, axis=-1))
        return DF



class SIFromMol:
    def __init__(self, ISI,step_max=None):
        self.ISI = ISI
        self.step_max = step_max
        self.n_atom_start = 4
        self.si, self.n_atom, self.n_adj = self.StrFromMol()
        self.Latom = self.si[:, 3].tolist()
        self.Sa, self.Sbon = self.StepBondFromMol()
        self.Ladj, self.Ladj_H, self.Ladj_nH, self.Lbon = CommonUtils.AdjacentLists(Latom=self.Latom,
                                                                                    n_atom=self.n_atom, Sa=self.Sa,
                                                                                    Sbon=self.Sbon)
        self.iH = np.where(np.array(self.Latom) == 'H')[0]
        self.SF = CommonUtils.FullStep(n_atom=self.n_atom, Sa=self.Sa, Ladj=self.Ladj,step_max=self.step_max)
        self.aro = CommonUtils.Aromatics(n_atom=self.n_atom, Lbon=self.Lbon, b_aro=1.5)
        self.Ladj_nl, self.lne = CommonUtils.DeleteEndAtom(Ladj_nH=self.Ladj_nH, Latom=self.Latom)
        self.cyc = CommonUtils.AtomInRing(SF=self.SF, Ladj_nl=self.Ladj_nl, lne=self.lne, Paro=self.aro)
        Lbon_str = CommonUtils.BondNumber2Line(Lbon=self.Lbon)
        self.Lbon_str = CommonUtils.BondLine2Curve(Lbon_str=Lbon_str, SF=self.SF, Ladj=self.Ladj, Ladj_nl=self.Ladj_nl,
                                                   lne=self.lne)


    def StrFromMol(self):
        """
        info from mol text
        :return: si(atom information,array)
                 n_atom(the number of atoms,int)
                 n_adj(the number of bonds,int)
        """
        ISI = copy.deepcopy(self.ISI)
        si = []
        n_atom = int(ISI[3][0:3])
        n_adj = int(ISI[3][3:6])
        for j in range(self.n_atom_start, self.n_atom_start + n_atom):
            si.append(ISI[j].split())
        si = np.array(si)
        return si, n_atom, n_adj

    def StepBondFromMol(self):
        """
        Get adjacency matrix (Sa) and bond  matrix (Sbon) from a mol file.
        :return: Sa and Sbon array
        """
        ISI = copy.deepcopy(self.ISI)
        Sa = np.zeros((self.n_atom, self.n_atom))
        Sbon = np.zeros((self.n_atom, self.n_atom))
        for j in range(self.n_atom_start + self.n_atom, self.n_atom_start + self.n_atom + self.n_adj):
            i_s = min([int(ISI[j][0:3]) - 1, int(ISI[j][3:6]) - 1])
            i_g = max([int(ISI[j][0:3]) - 1, int(ISI[j][3:6]) - 1])
            Sa[i_s, i_g] = 1
            Sbon[i_s, i_g] = int(ISI[j][6:9])
        Sa += Sa.T
        Sbon += Sbon.T
        Sbon[Sbon == 4] = 1.5
        return Sa, Sbon



class PositionProperty:
    @staticmethod
    def properties(name_atom):
        """
        the property matrices
        :param name_atom: Latom, list
        :return: P(property matrix, dict{P_key:array})
        """
        props = ['ele', 'wei', 'rad', 'noe', 'nes', 'ion']
        P = {}
        for prop in props:
            P[prop] = np.array([AtomicProperty[prop][atom] for atom in name_atom]).reshape(-1, 1)
        return P

    # 需要什么计算什么
    @staticmethod
    def filter_F(F, mode, val):
        """
        Filter the matrix F according to the given rule.
        :param F: ndarray,full step
        :param mode:str,
                - "eq": keep elements equal to `val`, set others to 0
                - "le": keep elements less than or equal to `val`, set others to 0
        :param val:float, threshold value for filtering.
        :return:
        """
        F_new = copy.deepcopy(F)
        if mode == "eq":
            F_new[F != val] = 0
        elif mode == "le":
            F_new[F > val] = 0
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return F_new

    @staticmethod
    def step(SI):
        """
        The step matrices
        :param SI: information from SIFromGjf or SIFromMol
        :return: a(the adjacent step matix,array),
                 b(the interphase step matix,array),
                 c(the jump step matix,array),
                 B(the adjacent and interphase step matix,array),
                 C(the adjacent interphase and jump step matix,array),
                 F(the full step matrix,array),
                 Caro_(sij<=3 and atom j is in an aromatic ring,array),
                 Ccyc_(sij<=3 and atom j is in a ring except for aromatic ring,array),
                 boncyc_(bij and atom j is in a ring except for aromatic ring,array),
                 FCDON_(sij and atom j is in a C=XN group, X is O or S, array),
                 FOSH_(sij and atom j is in a XH group, X is O or S,,array),
        """
        SI = copy.deepcopy(SI)
        a = SI.Sa
        F = SI.SF
        aro = SI.aro
        cyc = SI.cyc
        cyc = cyc - aro
        bon = SI.Sbon

        b = PositionProperty.filter_F(F=F, mode='eq', val=2)
        c = PositionProperty.filter_F(F=F, mode='eq', val=3)
        B = PositionProperty.filter_F(F=F, mode='le', val=2)
        C = PositionProperty.filter_F(F=F, mode='le', val=3)

        S = {'F': F,
             'a': a,
             'b': b,
             'c': c,
             'B': B,
             'C': C,
             'bon': bon,
             'Caro_': np.multiply(C, aro.T),
             'Ccyc_': np.multiply(C, cyc.T),
             'boncyc_': np.multiply(bon, cyc.T),
             }
        return S
