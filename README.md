# AdaptUnsatSMILES Enables the Discovery of High-Conductivity Ionic Liquids
# File Structure
- models: predict_EC.py and predict_Tm.py predict the electrical conductivity and melting point of ILs, respectively.
- py_common: The py_common directory contains the utility functions required by the prediction modules.
- DeNovoDesignTest.py: Validate de novo molecular design using the AdaptSMILES method.
- ProduceSMILES.py: The key program for implementing the AdaptSMILES approach.
- SMILESToMol.py: SMILES to Mol.
- StructureUnitsDeNovo.py: The configuration file for the AdaptSMILES design process.
- de_novo_design_test: JSON files contain the molecular structures, while XLSX files provide the PubChem benchmark results.
- ILs_design: Anion structures (JSON file), cation structures (JSON files), and results of ILs (JSON files).The dataset can be obtained from XXX.com.

# Requirements
- joblib==1.5.3
- networkx==3.6.1
- numpy==2.4.4
- pillow==12.2.0
- rdkit==2026.3.1
- requests==2.34.0
- scikit-learn==1.7.2
- scipy==1.17.1

# Funding Acknowledgements
This work was financially supported by the National Natural Science Foundation of China (22278319, 22478299, 22578332).

