program TestPascalFile;

var
  myInt: Integer;
  myFloat: Real;
  myBoolean: Boolean;
  myString: String;

begin
  myInt := 10;
  myFloat := 20.5;
  myBoolean := True;
  myString := 'Hello, World!';
  
  WriteLn('Integer value: ', myInt);
  WriteLn('Float value: ', myFloat:0:2);
  WriteLn('Boolean value: ', myBoolean);
  WriteLn('String value: ', myString);
end.
