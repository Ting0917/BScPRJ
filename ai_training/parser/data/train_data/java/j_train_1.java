import java.util.HashMap;
import java.util.Map;

public class TestJavaFile {
    
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complementary = target - nums[i];
            if (map.containsKey(complementary)) {
                return new int[] { i, map.get(complementary) };
            }
            map.put(nums[i], i);
        }
        return new int[] { -1, -1 };
    }

    public static void main(String[] args) {
        int[] nums = {2, 7, 11, 15};
        int target = 9;
        int[] result = twoSum(nums, target);
        System.out.println("Index 1: " + result[0] + ", Index 2: " + result[1]);
    }
}
