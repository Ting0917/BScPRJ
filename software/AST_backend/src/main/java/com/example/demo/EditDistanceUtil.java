package com.example.demo;

public class EditDistanceUtil {

    public static void main(String[] args) {
        System.out.println(EditDistanceUtil.computeEditDistance("kitten", "sitting")); // Expected: 3
        System.out.println(EditDistanceUtil.computeEditDistance("flaw", "lawn")); // Expected: 2
    }
    public static int computeEditDistance(String s1, String s2) {
        System.out.println(s1);
        System.out.println(s2);

        int output_len = s1.length();
        int target_len = s2.length();

        // Initialize the DP table
        int[][] OPT = new int[output_len + 1][target_len + 1];

        // Assign values to the first column
        for (int i = 1; i <= output_len; i++) {
            OPT[i][0] = i;
        }

        // Assign values to the first row
        for (int j = 1; j <= target_len; j++) {
            OPT[0][j] = j;
        }

        // Costs for insertions, deletions, and substitutions (alignments)
        int single_insert_cost = 1;
        int single_delete_cost = 1;
        int single_align_cost = 1;

        // Fill in the DP table
        for (int i = 1; i <= output_len; i++) {
            for (int j = 1; j <= target_len; j++) {
                int delta = (s1.charAt(i - 1) != s2.charAt(j - 1)) ? single_align_cost : 0;
                int alignment_cost = OPT[i - 1][j - 1] + delta;
                int delete_cost = OPT[i - 1][j] + single_delete_cost;
                int insertion_cost = OPT[i][j - 1] + single_insert_cost;

                OPT[i][j] = Math.min(Math.min(alignment_cost, delete_cost), insertion_cost);
            }
        }

        return OPT[output_len][target_len];
    }
}
