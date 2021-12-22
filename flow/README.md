# Flow-based molecule optimization
Code is based on [FastJTNNpy3](https://github.com/Bibyutatsu/FastJTNNpy3).

## JTVAE
Download model from [here](https://drive.google.com/file/d/1Ut1c_3kDBrKviM5IUGii2sqvwVHeARRP/view?usp=sharing) to `fast_molvae/save`.

## Dataset
Download zinc250k from [here](https://drive.google.com/file/d/1qr32WASlIIVIbTm4x8XXqZiH2HlqTq2M/view?usp=sharing) and unzip to `data/zinc250k`.

## Flow training
To train flow for logP optimization, run
```
python flow_train.py  --jtvae_path ../fast_molvae/save/model.epoch-19 --mol_path ../data/zinc250k/mol/ --vocab ../data/zinc250k/vocab.txt  --property_path ../data/zinc250k/logP/  --save_dir save_logP 
```

## Optimization (modification)
For now optimization is only possible for logP property.
To optimize logP run 
```
python evaluate_modification.py --flow_path save_logP/flow.epoch-50  --jtvae_path ../fast_molvae/save/model.epoch-19 --mol_path ../data/zinc250k/mol/ --vocab ../data/zinc250k/vocab.txt  --property_path ../data/zinc250k/logP/
```
Results will be saved in flow/optimization_results.

To visualize results call plot_hist function from flow/visualize.py.

## Optimization (generation)

