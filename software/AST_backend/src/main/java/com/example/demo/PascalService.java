package com.example.demo;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.*;

@Service
public class PascalService {

    private final HashGenerator hashGenerator;

    @Autowired
    public PascalService(HashGenerator hashGenerator) {
        this.hashGenerator = hashGenerator;
    }


    public String parsePascal(String code) {
        CharStream charStream = CharStreams.fromString(code);
        pascalLexer lexer = new pascalLexer(charStream);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        pascalParser parser = new pascalParser(tokens);

        // 自定義錯誤監聽器
        BaseErrorListener errorListener = new BaseErrorListener() {
            @Override
            public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol,
                                    int line, int charPositionInLine,
                                    String msg, RecognitionException e) {
                throw new IllegalStateException("Failed to parse at line " + line + " due to " + msg, e);
            }
        };

        // 將標準錯誤監聽器替換為自定義的錯誤監聽器
        parser.removeErrorListeners(); // 移除ANTLR的標準錯誤監聽器
        parser.addErrorListener(errorListener);

        try {
            ParseTree tree = parser.program();
            return tree.toStringTree(parser); // 無錯誤，返回解析樹
        } catch (IllegalStateException e) {
            return null; // 有語法錯誤，返回null
        }
    }

    public String toHash(String ast){
        return hashGenerator.generateHash(ast);
    }
}
