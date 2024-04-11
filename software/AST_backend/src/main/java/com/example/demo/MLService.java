package com.example.demo;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import com.fasterxml.jackson.databind.ObjectMapper;

@Service
public class MLService {

    private final RestTemplate restTemplate;
    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    public MLService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public String callGetJavaAstService(String pascalAst) {
        try {
            pascalAst = objectMapper.writeValueAsString(pascalAst);
        } catch (Exception e) {
            e.printStackTrace();
        }

        String requestBody = "{\"pascalAst\":" + pascalAst + "}";

//        String url = "http://localhost:5000/getjavaast";
//        String url = "http://host.docker.internal:5000/getjavaast";
        String url = "http://ml_backend:5000/getjavaast";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Object> request = new HttpEntity<>(requestBody, headers);

        // Using postForEntity since we're making a POST request.
        ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);
        return response.getBody();
    }
}
