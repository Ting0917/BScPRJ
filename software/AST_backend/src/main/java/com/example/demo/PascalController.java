package com.example.demo;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
@CrossOrigin
@RestController
public class PascalController {

    private final PascalService pascalService;
    private final CodeDataService codeDataService;

    private final MLService msService;

    @Autowired
    public PascalController(PascalService pascalService, CodeDataService codeDataService, MLService msService) {
        this.pascalService = pascalService;
        this.codeDataService = codeDataService;
        this.msService = msService;
    }


    @PostMapping("/processPascal")
    public String processPascalCode(@RequestBody PascalRequest request) {
        String code = request.getCode();

        String pascalAST = pascalService.parsePascal(code);

        if(pascalAST == null){
            return "Invalid Pascal";
        }
        System.out.println(pascalAST);

        String pascalASTHash = pascalService.toHash(pascalAST);

        if(codeDataService.isPascalASTHashInDatabase(pascalASTHash)){
            System.out.println("Bingo");
            return codeDataService.getJavaSourceCodeByPascalAstHash(pascalASTHash);
        }else{
            System.out.println("No bingo");
            String javaAst = msService.callGetJavaAstService(pascalAST);
            return codeDataService.findMostSimilarJavaSourceCode(javaAst);
        }
    }

    @PostMapping("/addCode")
    public CodeData addCodeData(@RequestBody CodeData codeData) {
        return codeDataService.saveCodeData(codeData);
    }

}
