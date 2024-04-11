# The parser Folder

This folder contains source code of data processing; typically refer to the Dataset chapter of the report.

Source code in this folder serves the following objectives:

Contains dataset
   dataset are all saved under ai_training/parser/data, folder structure is:
   /data/train_data/pascal
   /data/train_data/java
   /data/train_data/pascal_ast_str
   /data/train_data/java_ast_str

   /data/validation_data/pascal
   /data/validation_data/java
   /data/validation_data/pascal_ast_str
   /data/validation_data/java_ast_str

   /data/test_data/pascal
   /data/test_data/java
   /data/test_data/pascal_ast_str
   /data/test_data/java_ast_str

   /data/source_pascal_target_java_atom_test.json
   /data/source_pascal_target_java_train.json
   /data/source_pascal_target_java_validation.json

.

## To build Json files from Parse Tree

Build .json file in the desired format as Tree2Tree input, for training data, validation data, and atom test data.
build_json_from_parse_tree: build .json file as input of the Tree2Tree model
```bash
python build_json_from_parse_tree.py --folder test_data --result_file_name source_pascal_target_java_test
```

```bash
python build_json_from_parse_tree.py --folder train_data --result_file_name source_pascal_target_java_train
```

```bash
python build_json_from_parse_tree.py --folder validation_data --result_file_name source_pascal_target_java_validation
```

## Generate Input JSON file

```bash
python build_tree_dict.py
```
