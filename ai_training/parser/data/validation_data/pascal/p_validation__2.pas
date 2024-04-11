program TestPascalFile;
var
    i: Integer;
begin
    for i := 1 to 10 do
    begin
        if (i mod 2 = 0) then
            WriteLn(i, ' is even')
        else
            WriteLn(i, ' is odd');
    end;
end.
