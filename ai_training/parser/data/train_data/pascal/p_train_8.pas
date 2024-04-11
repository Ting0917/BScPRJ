program WhileLoopDemo;
var
    counter: Integer;
begin
    counter := 10;
    
    while counter > 0 do
    begin
        WriteLn(counter);
        counter := counter - 1; 
    end;
end.


