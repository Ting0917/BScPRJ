from flask import Flask, request, jsonify
from flask_cors import CORS

import build_json_from_parse_tree
import translate

app = Flask(__name__)
CORS(app)

@app.route('/getjavaast', methods=['POST'])
def getJavaAst():
    pascalAst = request.json['pascalAst']
    print(pascalAst)
    data = build_json_from_parse_tree.prepare_data(pascalAst)    
    java_ast = translate.main(data)
    
    return str(java_ast)

@app.route('/')
def home():
    return "Hello world"
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

