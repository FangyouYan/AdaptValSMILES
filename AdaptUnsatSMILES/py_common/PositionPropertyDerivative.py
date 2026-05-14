"""
Here is the finalised step matrices and property matrices for modelling.
"""
import numpy as np



def properties(P,label):
    """
    the finalised property matrices
    :param P: property matrices,dict
    :param label: data type ('P' or 'P_deH')
    :return: Data, DataType
    """
    Data = {'wei': P['wei'], 'ele': P['ele'], 'ion': P['ion'], 'rad': P['rad'],
            'noe': P['noe'], 'noe_nes': P['noe'] / P['nes'], 'bra': P['bra']}
    DataType = [label, '']
    return Data, DataType

def step(S,label):
    """
    the finalised step matrices
    :param S: step matrices,dict
    :param label: data type ('S' or 'S_deH')
    :return: Data, DataType
    """
    Data = {'F': S['F'], 'a': S['a'], 'B': S['B'], 'C': S['C'],
            'bon': S['bon'], 'Caro_': S['Caro_'],
            'Ccyc_': S['Ccyc_'], 'boncyc_': S['boncyc_'],
            }
    DataType = [label, '']
    return Data, DataType
