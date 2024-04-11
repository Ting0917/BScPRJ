@echo off
SETLOCAL

REM Set the path to your Python 3 installation if it's not already in the system PATH
REM For example, if Python is installed in "C:\Python39\", you might need to uncomment and adjust the line below:
REM set PATH=C:\Python39;%PATH%

REM Change directories to where the script is located if necessary
cd /d %~dp0

REM Executing the translation script with the specified parameters
python translate.py ^
--network tree2tree ^
--train_dir ..\model_ckpts\tree2tree\ ^
--input_format tree ^
--output_format tree ^
--num_epochs 100 ^
--batch_size 5 ^
--steps_per_checkpoint 5 ^
--train_data ..\..\parser\data\source_pascal_target_java_train.json ^
--val_data ..\..\parser\data\source_pascal_target_java_validation.json

ENDLOCAL
