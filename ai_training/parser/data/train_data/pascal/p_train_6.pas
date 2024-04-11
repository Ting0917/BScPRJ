program TestPascalFile;
var
    dayNumber: Integer;
    dayName: String;
begin
    Write('Enter a number (1-7) for the day of the week: ');
    ReadLn(dayNumber);
    
    case dayNumber of
        1: dayName := 'Monday';
        2: dayName := 'Tuesday';
        3: dayName := 'Wednesday';
        4: dayName := 'Thursday';
        5: dayName := 'Friday';
        6: dayName := 'Saturday';
        7: dayName := 'Sunday';
    else
        dayName := 'Invalid day';
    end;
    
    WriteLn('The day is: ', dayName);
end.
