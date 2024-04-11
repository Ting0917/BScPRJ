program TestPascalFile;
var
  result: Integer;

function AddNumbers(a, b: Integer): Integer;
begin
  AddNumbers := a + b;
end;

procedure DisplayResult(sum: Integer);
begin
  WriteLn('The sum is: ', sum);
end;

begin
  result := AddNumbers(10, 15);
  DisplayResult(result);
end.
