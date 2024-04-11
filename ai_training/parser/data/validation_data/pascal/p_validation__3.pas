program TestPascalFile;
var
  numbers: array[1..5] of Integer;
  sum, i: Integer;
begin
  numbers[1] := 1;
  numbers[2] := 2;
  numbers[3] := 3;
  numbers[4] := 4;
  numbers[5] := 5;

  sum := 0;
  for i := 1 to 5 do
    sum := sum + numbers[i];
  
  WriteLn('The sum is: ', sum);
end.
