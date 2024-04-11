program TestPascalFile;
var
  i, count: Integer;
begin
  Writeln('For loop (1 to 5):');
  for i := 1 to 5 do
    Writeln(i);

  Writeln('While loop (5 to 1):');
  count := 5;
  while count > 0 do
  begin
    Writeln(count);
    count := count - 1;
  end;
end.
