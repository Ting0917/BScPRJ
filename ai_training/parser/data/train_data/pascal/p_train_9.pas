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

    for i := Low(numbers) to High(numbers) do
    begin
        sum := sum + numbers[i];
    end;

    WriteLn('The sum of the array elements is: ', sum);
end.
