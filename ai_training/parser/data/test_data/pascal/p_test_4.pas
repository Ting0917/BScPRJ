program TestPascalFile;

type
  GradesArray = array[1..5] of Integer;
  Student = record
    name: String;
    grades: GradesArray;
  end;

var
  student1: Student;
  sum: Integer;
  i: Integer;
  average: Real;

begin
  student1.name := 'Selina';
  student1.grades[1] := 85;
  student1.grades[2] := 92;
  student1.grades[3] := 78;
  student1.grades[4] := 90;
  student1.grades[5] := 88;

  sum := 0;
  for i := 1 to 5 do
    sum := sum + student1.grades[i];
  
  average := sum / 5.0;
  WriteLn('Average grade for ', student1.name, ' is: ', average:0:2);
end.
