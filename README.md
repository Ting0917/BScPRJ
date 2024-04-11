# BSc Final Year Project: ML in Software Engineering- Legacy Programming Language Translation using ML approach

- GitHub Repository link: https://github.com/Ting0917/BScPRJ/tree/main
- Software URL: https://jpmfexmcd8.us-east-1.awsapprunner.com 

This repository includes the code implementation of Ting-Chen Chen's BSc Final Year Project, supervised by Dr.Kevin Lano.

The code implementation includes two main parts:

1. ai_training/
parser/ : Prepare data for the Tree2Tree Neural Network model, and all of the datasets are including in this folder.
tree2tree/ : The implementation of the Tree2Tree Neural Network
JavaToAst/ :Convert Java programs to AST.
PascalToAst/ :Convert Pascal programs to AST

Each folder contains its own README.md file for instructions and command lines.

Environment set up instructions(utilizing Python Virtual Environment) are provided clearly inside each folder where needed. All dependencies included.

2. software/
AST_backend/
database/
frontend/
ML_backend/

# Reference

1. The Tree2Tree model was proposed and  built by Chen et al. for their paper[[arXiv](https://arxiv.org/abs/1802.03691)][[NeurIPS](https://papers.nips.cc/paper/7521-tree-to-tree-neural-networks-for-program-translation)].


@inproceedings{chen2018tree,
  title={Tree-to-tree Neural Networks for Program Translation},
  author={Chen, Xinyun and Liu, Chang and Song, Dawn},
  booktitle={Proceedings of the 31st Advances in Neural Information Processing Systems},
  year={2018}
}


2. ANTLR4 is used to build TreeParser for Pascal and Java. ANTLR4 official website: https://www.antlr.org/

