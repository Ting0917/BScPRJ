package com.example.demo;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "Code")
public class CodeData {
    @Id
    private String pascalAstHash;
    private String javaSourceCode;
    private String javaAst;

    // 构造函数
    public CodeData(String pascalAstHash, String javaSourceCode, String javaAst) {
        this.pascalAstHash = pascalAstHash;
        this.javaSourceCode = javaSourceCode;
        this.javaAst = javaAst;
    }

    String getPascal(){
        return pascalAstHash;
    }

    void setPascal(String pascalAstHash){
        this.pascalAstHash = pascalAstHash;
    }

    public String getJavaSourceCode() {
        return javaSourceCode;
    }

    public String getJavaAst(){
        return this.javaAst;
    }

}
