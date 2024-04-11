package com.example.demo;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException; // Import IOException

@SpringBootApplication
public class DemoApplication {

	public static void main(String[] args) {
		try {
			// Try to read the Pascal source code
			CharStream charStream = CharStreams.fromFileName("D:/part-time/UK/AST/pascal/src/main/java/org/example/TestPascalFile.pas");

			// Create the lexer
			pascalLexer lexer = new pascalLexer(charStream);
			CommonTokenStream tokens = new CommonTokenStream(lexer);

			// Create the parser
			pascalParser parser = new pascalParser(tokens);

			// Parse the code and generate the AST
			ParseTree tree = parser.program(); // Assuming 'program' is the starting rule of the grammar

			// Print the AST
			System.out.println(tree.toStringTree(parser));
		} catch (IOException e) {
			e.printStackTrace(); // Handle the exception here, maybe log it or inform the user
		}

		SpringApplication.run(DemoApplication.class, args);
	}
}
