import requests
import os
from glob import glob
from pathlib import Path
import re

def send_data(url, data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    return response


# Base directory where the 'train', 'test', 'evaluation' folders are located
base_dir = '../../ai_training/parser/data'  # Update this to your specific path

# Define the folders and corresponding variables
folders_to_variables = {
    'pascal_ast_str': 'pascal_ast_hash',
    'java': 'java_source_code',
    'java_ast_str': 'java_ast'
}

# This function will read a .txt file and return its content
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content.rstrip('\n')


# Helper function to extract the number from the file name
def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else 0


# This function will process each directory and apply a function to each file, sorting them by number in the file name
def process_directories(base_dir, data_types, process_function):
    for data_type in data_types:
        dic = []
        for folder, variable in folders_to_variables.items():
            directory = Path(base_dir) / data_type / folder
            print(directory)
            files = list(directory.glob('*.txt')) + list(directory.glob('*.java'))
            sorted_files = sorted(files, key=lambda x: extract_number(x.stem))
            tmp = []
            for txt_file in sorted_files:
                tmp.append(txt_file)
            dic.append(tmp)
        print(dic)

        for j in range(len(dic[0])):        
            ok = {}
            for i in range(len(dic)):
                content = process_function(dic[i][j])
                if i == 0:
                    ok["pascalAstHash"] =  content
                elif i==1:
                    ok["javaSourceCode"] = content
                else:
                    ok["javaAst"] =  content
                print(dic[i][j])
        
            print(ok)
            response = send_data('http://localhost:8080/addCode', ok)
            print(f'Response Status Code: {response.status_code}, Response Text: {response.text}')

# Directories to process
data_types = ['train_data', 'test_data', 'validation_data']

# Process the directories and print the content of each file
process_directories(base_dir, data_types, read_txt_file)
