package org.example;

import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;
import java.io.IOException;


public class Main {
    public static void main(String[] args) {


        try {
            // 读取 Java 源代码
            CharStream charStream = CharStreams.fromFileName("C:/Users/selin/FinalProject/JavaToAst/src/main/java/org/example/TestJavaFile.java");

            // 创建词法分析器
            JavaLexer lexer = new JavaLexer(charStream);
            CommonTokenStream tokens = new CommonTokenStream(lexer);

            // 创建语法解析器
            JavaParser parser = new JavaParser(tokens);

            // 解析代码并生成 AST
            ParseTree tree = parser.compilationUnit();

            System.out.println("AST Tree");
            // 输出 AST
            System.out.println(tree.toStringTree(parser));

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}