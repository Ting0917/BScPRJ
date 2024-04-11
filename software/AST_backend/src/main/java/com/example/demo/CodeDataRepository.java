package com.example.demo;
import com.example.demo.CodeData;

import org.springframework.data.mongodb.repository.MongoRepository;
import java.util.Optional;
public interface CodeDataRepository extends MongoRepository<CodeData, String> {
    Optional<CodeData> findByPascalAstHash(String pascalAstHash);
}
