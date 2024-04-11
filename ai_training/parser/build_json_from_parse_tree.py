"""
Author: Ting-Chen Chen

This script is designed to convert string-formatted parse trees into dictionary representations of ASTs (Abstract Syntax Trees). 
It supports processing for training, validation, and atom test data sets. The script expects parse trees to be stored 
in specific directories and outputs the resulting ASTs as JSON files. These JSON files are intended for use in 
further processing or machine learning models that require structured tree input.

Usage Examples:
# For processing training data and saving the output in the default JSON file.
python build_json_from_parse_tree.py

# For processing validation data, specifying the input folder and the output JSON file name.
python build_json_from_parse_tree.py --folder validation_data --result_file_name source_py_target_js_validation

# For processing atom test data, specifying the input folder and the output JSON file name.
python build_json_from_parse_tree.py --folder test_data --result_file_name source_py_target_js_atom_test
"""

import os
import json
import argparse

parser = argparse.ArgumentParser(description='provide me detail of which parse tree string(s) to be converted into JSON file.')
parser.add_argument('--folder',default='train_data',
help="The folder under /data/ which contrains string-formated parse tree .txt files, generated from raw programs using ANTLR4.(default:train_data)")
parser.add_argument('--result_file_name', default='source_pascal_target_java_train',
help="The name of the resulted json file which will then be used as input data of the Tree2Tree model. Do NOT add the .json postfix. The file will be placed under the root directory.")
parser.add_argument('--source', default='pascal',choices=['pascal','java'],
help="The source language")
parser.add_argument('--example', action='store_true',
help="Whether use this program to generate input JSON file for the Tree2Tree model (default False), or use this program to generate a JSON file for one example string-formatted parse tree .txt file (manual set this param to be True.)")
parser.add_argument('--example_file', default='example_js_ast_str.txt',
help="Build JSON file for one example string-formatted parse tree .txt file. (default is 'example_js_ast_str.txt')")
args = parser.parse_args()

def build_ast(token_list):
    """
    Converts a list of tokens representing a parse tree into a dictionary structure representing the AST
    
    param token_list: List of tokens from the parse tree.
    return: A dictionary representing the AST.
    """
    stack = []
    for token in token_list:
        if token == '(':
            stack.append(token)
        elif token == ')':
            inside_parenthese = []
            cur = stack.pop()   
            while cur != '(':
                inside_parenthese.append(cur)
                cur = stack.pop()

            if not inside_parenthese:
                    # Special case: '(' and ')' are the tokens themselves with no content between them.
                left_parenthesis_dict = {'root': '(', 'children': []}
                right_parenthesis_dict = {'root': ')', 'children': []}
                # Create a special structure to hold both parentheses as siblings
                # This assumes your AST format allows for this kind of structure.
                # Alternatively, adjust this part according to your specific AST requirements.
                parentheses_dict = {'root': 'parentheses', 'children': [left_parenthesis_dict, right_parenthesis_dict]}
                stack.append(parentheses_dict)
                continue  # Continue to the next token
            
            inside_parenthese.reverse()
            root = inside_parenthese.pop(0) 
            children = inside_parenthese # may be empty or many children
            root_dict = {'root': root, 'children':[]}
            for child in children:
                if isinstance(child, dict):
                    root_dict['children'].append(child)
                elif isinstance(child, str):
                    child_dict = {'root':child, 'children':[]}
                    root_dict['children'].append(child_dict)
            stack.append(root_dict)
        else:
            stack.append(token)
    ast_dict = stack[0]
    return ast_dict

def get_token_list(ast_string, file_name):
    """
    Processes a string-formatted parse tree into a list of tokens.
    
    param ast_string: The string representation of the AST.
    param file_name: The file name containing the AST string (unused in the function).
    return: A list of tokens.
    """
    ast_string = ast_string.replace('(',' ( ')
    ast_string = ast_string.replace(')',' ) ')
    token_list = ast_string.split()
    return token_list

def write_one_json():
    """
    Generates a JSON file for one example string-formatted parse tree.
    """
    file_name = args.example_file
    file = open(file_name, 'r')
    file_str = file.read()
    token_list = get_token_list(file_str, file_name)
    dict_ast = build_ast(token_list)
    result_file_dir = file_name[:-3]+'json'
    result_file = open(result_file_dir, 'w')
    result_file.write(json.dumps(dict_ast))
    result_file.close()
    print('Example JSON file is written.')

def prepare_data():
    """
    Prepares and generates JSON files from the provided parse trees for training, validation, or test data.
    """
    folder_under_data = args.folder
    data_type = folder_under_data[:-5]
    print(folder_under_data)
    count = 1
    root_dir = os.getcwd()
    pascal_file_name = root_dir+'/data/'+folder_under_data+'/pascal_ast_str/p_'+data_type+'_'+str(count)+'_ast_str.txt'
    java_file_name = root_dir+'/data/'+folder_under_data+'/java_ast_str/j_'+data_type+'_'+str(count)+'_ast_str.txt'
    pascal_file = open(pascal_file_name, 'r')
    java_file = open(java_file_name, 'r')
    result = []
    while True:
        # Only fill in target_ast and source_ast, ignore source_prog and target_prog
        one_data = {"source_prog":None, "target_prog":None, 
                    "source_ast":None, "target_ast":None}
        pascal_str = pascal_file.read()
        java_str = java_file.read()
        
        pascal_file_name = get_token_list(pascal_str, pascal_file_name)
        print(f"Num = {count}, Start to process pascal")
        pascal_dict = build_ast(pascal_file_name)
  
        java_token_list = get_token_list(java_str, java_file_name)

        print(f"Num = {count}, Start to process java")
        java_dict = build_ast(java_token_list)
        if len(pascal_file_name)<=4 or len(java_token_list)<=4:
            break
        if args.source == 'pascal':
            one_data["source_ast"] = pascal_dict
            one_data["target_ast"] = java_dict
            target_langauge = 'java'
        else:
            one_data["source_ast"] = java_dict
            one_data["target_ast"] = pascal_dict
            target_langauge = 'pascal'
        result.append(one_data)
        print('Generated json obj num:', count)
        count +=1
        pascal_file_name = root_dir+'/data/'+folder_under_data+'/pascal_ast_str/p_'+data_type+'_'+str(count)+'_ast_str.txt'
        java_file_name = root_dir+'/data/'+folder_under_data+'/java_ast_str/j_'+data_type+'_'+str(count)+'_ast_str.txt'
        
        if os.path.exists(pascal_file_name) and os.path.exists(java_file_name):
            pascal_file = open(pascal_file_name, 'r')
            java_file = open(java_file_name, 'r')
        else:
            print('stopped at not-exist file:', pascal_file_name)
            break
        
    if args.result_file_name is None:
        result_file_name = 'source_'+args.source+'_target_'+target_langauge+folder_under_data
    else:
        result_file_name = args.result_file_name
    result_file_name = root_dir +'/data/'+result_file_name+'.json'
    result_file = open(result_file_name, 'w')
    result_file.write(json.dumps(result))
    print('FINISHED')

if __name__ == '__main__':
    if args.example:
        write_one_json()
    else:
        prepare_data()
    
