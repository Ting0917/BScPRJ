package org.example;

import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;
import java.io.IOException;


public class Main {
    public static void main(String[] args) {

        try {
            // read the original pascal code
            CharStream charStream = CharStreams.fromFileName("C:/Users/selin/FinalProject/PascalToAst/src/main/java/org/example/TestPascalFile.pas");

            // translates to "Create a lexical analyzer"
            pascalLexer lexer = new pascalLexer(charStream);
            CommonTokenStream tokens = new CommonTokenStream(lexer);

            // =translates to "Create a syntax parser
            pascalParser parser = new pascalParser(tokens);

            // create AST tree here
            ParseTree tree = parser.program(); // Assume 'program' is the beginning

            // print out the result
            System.out.println(tree.toStringTree(parser));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}