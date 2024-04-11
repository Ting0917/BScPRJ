# The tree2tree Folder

This folder provides code implementation of the Tree2Tree model [[arXiv](https://arxiv.org/abs/1802.03691)][[NeurIPS](https://papers.nips.cc/paper/7521-tree-to-tree-neural-networks-for-program-translation)], refers to the Model Design and Implementation chapter in the report.

This folder follows the following file structure:

1. model_ckpts/tree2tree/: where the trained model will be saved
2. result/: where the translation result outputed from a trained Tree2Tree model will be saved.
3. utils/dataset_splitter/: a simple Python script that can split a large json file to a smaller one.
4. src/: the source code of the Tree2Tree implementation
   4.1 src/translate.py: the main control file of the Tree2Tree model. All interactions from user to the model starts from this file.
   4.2 src/network.py: contains model design. specifically, the TreeEncoder and Tree2TreeModel, used in this project.
   4.3 src/Tree.py: contains classes defining Tree structure. Specifically, the BinaryTree and TreeManger, used in this project.
   4.4 src/data_utils.py: contains data processing functions.

# Citation

The code implementation in this folder was initially built by Chent et al. and published with their paper; commented and adjusted by Ting-Chen Chen as BSc final year project (PRJ) at King's College London, supervised by Kevin Lano.

Chen et al. Paper Citation:

```
@inproceedings{chen2018tree,
  title={Tree-to-tree Neural Networks for Program Translation},
  author={Chen, Xinyun and Liu, Chang and Song, Dawn},
  booktitle={Proceedings of the 31st Advances in Neural Information Processing Systems},
  year={2018}
}
```

# Prerequisites

check `requirements.txt`.


## dataset for Ting-Chen Chen's Final Project

/parser/data/train_data
/parser/data/validation_data
/parser/data/test_data


## Run experiments

In the following we list some important arguments in `translate.py`:

- `--train_data`, `--val_data`, `--test_data`: path to the preprocessed dataset.
- `--load_model`: path to the pretrained model (optional).
- `--train_dir`: path to the folder to save the model checkpoints.
- `--input_format`, `--output_format`: can be chosen from `seq` (tokenized sequential program) and `tree` (parse tree).
- `--test`: add this command during the test time, and remember to set `--load_model` during evaluation.

## Train the model

```bash
python translate.py --network tree2tree --train_dir ../model_ckpts/tree2tree/ --input_format tree --output_format tree
```

#python3

```bash
python3 translate.py 
--network tree2tree 
--train_dir ../model_ckpts/tree2tree/ 
--input_format tree 
--output_format tree
```

```bash
python3 translate.py 
--network tree2tree 
--train_dir ../model_ckpts/tree2tree/ 
--input_format tree --output_format tree 
--num_epochs 100 --batch_size 5 
--steps_per_checkpoint 5 
--train_data ../../parser/data/source_pascal_target_java_train.json 
--val_data ../../parser/data/source_pascal_target_java_validation.json
```


### train
For Windows:

```bash
python translate.py 
--network tree2tree 
--train_dir ..\model_ckpts\tree2tree\ --input_format tree 
--output_format tree 
--num_epochs 100 
--batch_size 5 
--steps_per_checkpoint 5 
--train_data ..\..\parser\data\source_pascal_target_java_train.json 
--val_data ..\..\parser\data\source_pascal_target_java_validation.json

```
For Linux:

```bash
python translate.py --network tree2tree --train_dir ../model_ckpts/tree2tree/ --input_format tree --output_format tree --num_epochs 100 --batch_size 5   --steps_per_checkpoint 5 --train_data ../../parser/data/source_pascal_target_java_train.json --val_data ../../parser/data/source_pascal_target_java_validation.json
```

### Test
For Linux and Windows

```bash Best Loss
python translate.py --network tree2tree --test --load_model ../model_ckpts/tree2tree/best_loss_translate_195.ckpt --train_data ../../parser/data/source_pascal_target_java_train.json --test_data ../../parser/data/source_pascal_target_java_test.json --input_format tree --output_format tree
```

```bash Best Eval Loss
python translate.py --network tree2tree --test --load_model ../model_ckpts/tree2tree/best_eval_loss_translate_195.ckpt --train_data ../../parser/data/source_pascal_target_java_train.json --test_data ../../parser/data/source_pascal_target_java_test.json --input_format tree --output_format tree
```


The environment of this project is managed by python virtual machine with pip.
Required Python packages and version info are written in the 'requirement.txt' file.To build the  environment:

```bash
cd tree2tree
```

create a python virtual machine (if not have one already)

```bash
python -m venv virtual_environment_name
```

activate your python virtual machine

For Windows cmd
```bash
virtual_environment_name\Scripts\activate
```

For Windows powershell

```bash
.\virtual_environment_name\Scripts\Activate.ps1
```

For MacOs
```bash
source virtual_environment_name/bin/activate
```
install the packages with desired version as recorded in the requirements.txt file

```bash
pip3 install -r requirements.txt
```
```bash
pip install -r requirements.txt
```

