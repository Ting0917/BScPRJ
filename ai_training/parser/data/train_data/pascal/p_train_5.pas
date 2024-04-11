program TestPascalFile;
var
    number: Integer;
begin
    Write('Enter an integer: ');
    ReadLn(number);
    
    if number > 0 then
        WriteLn('The number is positive.')
    else if number < 0 then
        WriteLn('The number is negative.')
    else
        WriteLn('The number is zero.');
end.
