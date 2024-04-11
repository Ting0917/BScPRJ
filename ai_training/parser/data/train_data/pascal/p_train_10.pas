program TestPascalFile;
var
    firstPart, secondPart, combinedString: String;
    sub: String;
    length: Integer;
begin
    firstPart := 'Hello, ';
    secondPart := 'World!';
    combinedString := firstPart + secondPart;
    WriteLn('Concatenated String: ', combinedString);
    
    sub := Copy(combinedString, 8, 5);
    WriteLn('Extracted Substring: ', sub);
    
    length := Length(combinedString);
    WriteLn('Length of String: ', length);
end.



