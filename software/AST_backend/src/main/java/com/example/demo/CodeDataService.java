package com.example.demo;

import com.example.demo.CodeData;
import com.example.demo.CodeDataRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class CodeDataService {

    private final HashGenerator hashGenerator;
    private final CodeDataRepository codeDataRepository;

    @Autowired
    public CodeDataService(CodeDataRepository codeDataRepository,
                           HashGenerator hashGenerator
                           ) {
        this.codeDataRepository = codeDataRepository;
        this.hashGenerator = hashGenerator;
    }

    public CodeData saveCodeData(CodeData codeData) {
        System.out.println(codeData.getPascal());
        String hash = hashGenerator.generateHash(codeData.getPascal());
        codeData.setPascal(hash);
        System.out.println(hash);
        return codeDataRepository.save(codeData);
    }

    public boolean isPascalASTHashInDatabase(String hash){
        return codeDataRepository.findByPascalAstHash(hash).isPresent();
    }

    public String getJavaSourceCodeByPascalAstHash(String pascalAstHash) {
        Optional<CodeData> codeDataOptional = codeDataRepository.findByPascalAstHash(pascalAstHash);
        if (codeDataOptional.isPresent()) {
            return codeDataOptional.get().getJavaSourceCode();
        } else {
            return null; // Or any other appropriate value or exception handling
        }
    }

    public String findMostSimilarJavaSourceCode(String javaAst) {
        List<CodeData> allCodeData = codeDataRepository.findAll();
        CodeData mostSimilarCodeData = null;
        int smallestEditDistance = Integer.MAX_VALUE;

        for (CodeData codeData : allCodeData) {
            int currentDistance = EditDistanceUtil.computeEditDistance(javaAst, codeData.getJavaAst());
            System.out.println(currentDistance);
            if (currentDistance < smallestEditDistance) {
                smallestEditDistance = currentDistance;
                mostSimilarCodeData = codeData;
            }
        }

        return mostSimilarCodeData != null ? mostSimilarCodeData.getJavaSourceCode() : null;
    }

}
