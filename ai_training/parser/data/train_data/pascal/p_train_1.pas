program TestPascalFile;

type
    TIntArray = array of Integer;

function TwoSum(nums: TIntArray; target: Integer): TIntArray;
var
    map: array of Integer;
    i, complementary, index: Integer;
begin
    SetLength(map, 10000);
    for i := 0 to High(nums) do
    begin
        complementary := target - nums[i];
        if (map[complementary] <> 0) then
        begin
            SetLength(Result, 2);
            Result[0] := i + 1;
            Result[1] := map[complementary];
            Exit;
        end;
        map[nums[i]] := i + 1;
    end;
    SetLength(Result, 2);
    Result[0] := -1;
    Result[1] := -1;
end;

var
    nums: TIntArray;
    result: TIntArray;
begin
    SetLength(nums, 4);
    nums[0] := 2;
    nums[1] := 7;
    nums[2] := 11;
    nums[3] := 15;
    result := TwoSum(nums, 9);
    WriteLn('Index 1: ', result[0], ' Index 2: ', result[1]);
end.