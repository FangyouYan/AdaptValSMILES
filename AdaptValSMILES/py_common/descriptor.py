import numpy as np

class MolecularInformation:
    @staticmethod
    def InformationFromStep(Latom, Mwei, SF_H,calc_keys):
        """
        iformation from a step
        :param Latom: atom type list
        :param Mwei: molecular weight Pwei
        :param SF_H:full step matrix without the hydrogen atom
        :param calc_keys:
        :return:if_s(the number of atoms, the number of heavy atoms,molecular weight,
                    the max step with SF_H, the sum of SF_H,
                    array)
        """

        feature_map = {
            "n_atom": lambda: len(Latom),
            "n_nonH": lambda: len(Latom) - np.sum(np.array(Latom) == 'H'),
            "mw": lambda: np.sum(abs(Mwei), axis=0)[0],
            "st_m": lambda: np.max(SF_H),
            "st_s": lambda: np.sum(SF_H),
        }
        if_s = [feature_map[k]() for k in calc_keys if k in feature_map]

        return np.array(if_s).reshape(1,-1)


