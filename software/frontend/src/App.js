import React, { useState } from 'react';
import './App.css';
import { Card, CardContent, Typography, Box, Container, TextField, Button } from '@mui/material';


function App() {
  const [pascalCode, setPascalCode] = useState('');
  const [javaOutput, setJavaOutput] = useState('');

  const handlePascalInputChange = (event) => {
    setPascalCode(event.target.value);
  };

  const processPascalCode = () => {
    console.log(JSON.stringify({ code: pascalCode }));
  
    fetch('http://localhost:8080/processPascal', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code: pascalCode }),
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.text(); 
    })
    .then(text => {
      setJavaOutput(text); 
    })
    .catch(error => {
      console.error('Error:', error);
      setJavaOutput('Error processing Pascal code.'); 
    });
  };
  
  

  return (
    <div className="App">
     
     <div className="headerDiv">
      <Typography  variant="h2" component="h2">
      Selina's Pascal to Java Transcoder
        </Typography>
        </div>

      <Container className="cardContainer">
        <Box className="cardBox">
          <Card className="mainCard" elevation={0}>
            <CardContent className="cardContent">
              <Typography variant="h5" component="h2">
                 Pascal(Input)
              </Typography>
              <div className="textFieldDiv">
              <TextField
                className="inputField"
                multiline
                rows={20}
                placeholder="Please enter Pascal code"
                variant="outlined"
                fullWidth
                value={pascalCode}
                onChange={handlePascalInputChange}
                InputProps={{
                  notchedOutline: {
                    borderColor: 'black',
                  },
                }}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&.Mui-focused fieldset': {
                      borderColor: 'black', 
                    },
                  },
                }}
              />
              </div>
            </CardContent>
          </Card>
          

          <Card className="mainCard" elevation={0}>
            <CardContent  className="cardContent">
              <Typography variant="h5" component="h2">
                Java(Output)
              </Typography>
              <div className="textFieldDiv">
              <TextField
                className="outputField"
                multiline
                rows={20}
                placeholder="Java output will be displayed here"
                variant="outlined"
                fullWidth
                value={javaOutput}
                InputProps={{
                  readOnly: true,
                }}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&.Mui-focused fieldset': {
                      borderColor: 'black', 
                    },
                  },
                }}
              />
              </div>
            </CardContent>
          </Card>
        </Box>
      </Container>

      <Button variant="contained" color="inherit" onClick={processPascalCode}  style={{ marginTop: '20px'}}>
            Trans Code
          </Button>
    </div>
  );
}

export default App;
