# Flow-based molecule optimization
Code is based on [FastJTNNpy3](https://github.com/Bibyutatsu/FastJTNNpy3).

## JTVAE
Download model from [here](https://drive.google.com/file/d/1Ut1c_3kDBrKviM5IUGii2sqvwVHeARRP/view?usp=sharing) to `fast_molvae/save`.

## Dataset
Download zinc250k from [here](https://drive.google.com/file/d/1qr32WASlIIVIbTm4x8XXqZiH2HlqTq2M/view?usp=sharing) and unzip to `data/zinc250k`.

## Flow training
To train flow for logP optimization, run
```
python flow_train.py  --jtvae_path ../fast_molvae/save/model.epoch-19 --mol_path ../data/zinc250k/mol/ --vocab ../data/zinc250k/vocab.txt  --property_path ../data/zinc250k/logP/  --save_dir save_logP --flow_type NICE 
```

### CNF training
To train CNF for logP optimization, run
```
python flow_train.py  --jtvae ../fast_molvae/save/model.epoch-19 --mol_path ../data/zinc250k/mol/ --property_path ../data/zinc250k/logP/ --vocab ../data/zinc250k/vocab.txt  --save_dir save_cnf_logP_b1  --flow_type CNF --epochs 10 --flow_sigma_decay 0.8 --flow_n_blocks 1
```

## Optimization (modification)
Run 
```
python evaluate_modification.py --flow_path save_realnvp_cond_logP/flow.epoch-100  --jtvae_path ../fast_molvae/save/model.epoch-19 --mol_path ../data/zinc250k/mol/ --vocab ../data/zinc250k/vocab.txt  --property_path ../data/zinc250k/logP/ --values -2.0 2.0 --flow_type "RealNVP" --conditional
```
Results will be saved in flow/optimization_results.

To visualize results use flow/visualize.py.

## Optimization (generation)

```
python evaluate_modification.py --flow_path save_realnvp_cond_logP/flow.epoch-100  --jtvae_path ../fast_molvae/save/model.epoch-19 --mol_path ../data/zinc250k/mol/ --vocab ../data/zinc250k/vocab.txt  --property_path ../data/zinc250k/logP/ --values -2.0 2.0 --flow_type "RealNVP" --conditional --generate --generate_sigma 0.5
```
