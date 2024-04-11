program TestPascalFile;
var
    myInt: Integer;
    myString: String;
begin
    Write('Enter an integer: ');
    ReadLn(myInt);
    
    Write('Enter a string: ');
    ReadLn(myString);
    
    WriteLn('You entered integer: ', myInt);
    WriteLn('You entered string: ', myString);
end.
