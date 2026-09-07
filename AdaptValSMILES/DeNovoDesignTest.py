import json
import time

from AdaptValSMILES import produce_SMILES

N2_N1 = {
    '1': 3, '2': 10, '3': 32, '4': 55, '5': 200,
    '6':500,'7':700,'8':1500,'9':3000,'10':4000,
    '11':4000,'12':4000,'13':4000,'14':4000,'15':4000,
    '16':4000,'17':4000,'18':4000,'19':4000,'20':4000,
    '21':5000,'22':5000,'23':5000,'24':5000,'25':5000,
    '26':5000,'27':5000,'28':5000,'29':5000,'30':5000,
}


cyc_num_max = 0
SMILES_R = {}

N2_N1_keys = list(N2_N1.keys())
N2_N1_keys.sort(key=int)

t0=time.time()
for N2_i in N2_N1_keys:
    t1 = time.time()
    N1_i = N2_N1[N2_i]
    SMILES_i = produce_SMILES(N1=N1_i, N2=int(N2_i), cyc_max=0, shared_bond_prob=0.05, use_scaffold=False)
    SMILES_R[N2_i] = SMILES_i
    t2 = time.time()
    print(N2_i,t2-t1)
print(t2-t0)

with open('./de_novo_design_test/De_SMILES.json','w',encoding='utf-8') as f:
    json.dump(SMILES_R, f, ensure_ascii=False, indent=4)