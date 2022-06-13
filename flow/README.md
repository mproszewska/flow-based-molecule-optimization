# Flow-based molecule optimization
Code is based on [FastJTNNpy3](https://github.com/Bibyutatsu/FastJTNNpy3).

## JTVAE
Download model from [here](https://drive.google.com/file/d/1Ut1c_3kDBrKviM5IUGii2sqvwVHeARRP/view?usp=sharing) to `fast_molvae/save`.

## Dataset
Download zinc250k from [here](https://drive.google.com/file/d/1qr32WASlIIVIbTm4x8XXqZiH2HlqTq2M/view?usp=sharing) and unzip to `data/zinc250k`.

## Flow training
To train flow for logP optimization, run
```
python flow_train.py  --jtvae_path ../fast_molvae/save/model.epoch-19 --mol_path ../data/zinc250k/mol/ --vocab ../data/zinc250k/vocab.txt  --attr_path ../data/zinc250k/logP/  --save_dir save_logP --flow_type NICE --scaffold_path zinc250k_220323/scaffold_one_hot/ --flow_use_logvar --encoder_a_identity --smiles_path zinc250k_220323/smiles/
```

### CNF training
To train CNF for logP optimization, run
```
python flow_train.py  --jtvae ../fast_molvae/save/model.epoch-19 --mol_path ../data/zinc250k/mol/ --attr_path ../data/zinc250k/logP/ --vocab ../data/zinc250k/vocab.txt  --save_dir save_cnf_logP_b1  --flow_type CNF --epochs 10 --flow_sigma_decay 0.8 --flow_n_blocks 1 --scaffold_path zinc250k_220323/scaffold_one_hot/ --flow_use_logvar --encoder_a_identity --smiles_path zinc250k_220323/smiles/
```

## Optimization (modification)
Run 
``` 
python flow_evaluate.py --flow_path save_logP/flow.epoch-100  --jtvae_path ../fast_molvae/save/model.epoch-19 --mol_path ../data/zinc250k/mol/ --vocab ../data/zinc250k/vocab.txt  --attr_path ../data/zinc250k/logP/ --values 1.5 3.0 4.5 --flow_type NICE --scaffold_path zinc250k_220323/scaffold_one_hot/ --flow_use_logvar --encoder_a_identity --smiles_path zinc250k_220323/smiles/
```

To visualize results use flow/visualize.py.

## Optimization (generation)

```

python flow_evaluate.py --flow_path save_logP/flow.epoch-100  --jtvae_path ../fast_molvae/save/model.epoch-19 --mol_path ../data/zinc250k/mol/ --vocab ../data/zinc250k/vocab.txt  --attr_path ../data/zinc250k/logP/ --values 1.5 3.0 4.5 --flow_type NICE --scaffold_path zinc250k_220323/scaffold_one_hot/ --flow_use_logvar --encoder_a_identity --smiles_path zinc250k_220323/smiles/ --generate flow --generate_sigma 0.5
```

## Preprocess dataset

```
obabel -ismi ../data/zinc250k/smiles.smi -s'C(F)(F)F' -O CF3_smiles.smi
obabel -ismi ../data/zinc250k/smiles.smi -s'[!#1]Cl' -osmi -O Cl_smiles.smi
obabel -ismi ../data/zinc250k/smiles.smi -s'[!#1]F' -osmi -O F_smiles.smi
obabel -ismi ../data/zinc250k/smiles.smi -s'[!#1]I' -osmi -O I_smiles.smi
obabel -ismi ../data/zinc250k/smiles.smi -s'C#N' -osmi -O CN_smiles.smi
```
