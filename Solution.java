import java.util.HashMap;
import java.util.Map;

class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

public class Solution {

    // Q1
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i]))
                return new int[] { i, map.get(target - nums[i]) };
            map.put(nums[i], i);
        }
        return null;
    }

    // Q2
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode();
        ListNode cursor = dummy;
        int carryFlag = 0, x, y;

        while (l1 != null || l2 != null || carryFlag != 0) {
            if (l1 != null) {
                x = l1.val;
                l1 = l1.next;
            } else
                x = 0;

            if (l2 != null) {
                y = l2.val;
                l2 = l2.next;
            } else
                y = 0;

            cursor.next = new ListNode((x + y + carryFlag) % 10);
            cursor = cursor.next;
            carryFlag = (x + y + carryFlag) / 10;
        }
        return dummy.next;
    }

    // Q3
    public int lengthOfLongestSubstring(String s) {
        HashMap<Character, Integer> map = new HashMap<>();
        int result = 0, left = 0;

        for (int right = 0; right < s.length(); right++) {
            Character c = s.charAt(right);
            if (map.containsKey(c))
                left = Math.max(left, map.get(c) + 1);
            map.put(c, right);
            result = Math.max(result, right - left + 1);
        }
        return result;
    }

    // Q5
    public String longestPalindrome(String s) {
        int[] res = { 0, 0 };

        for (int i = 0; i < s.length(); i++) {
            int[] res1 = longestPalindrome_helper(s, i, i);
            int[] res2 = longestPalindrome_helper(s, i, i + 1);
            if ((res1[1] - res1[0]) > res[1] - res[0])
                res = res1;
            if ((res2[1] - res2[0]) > res[1] - res[0])
                res = res2;
        }
        return s.substring(res[0], res[1] + 1);
    }

    int[] longestPalindrome_helper(String s, int left, int right) {
        // Expand around center
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return new int[] { left + 1, right - 1 };
    }

    // Q6
    public String convert(String s, int numRows) {
        if (numRows == 1)
            return s;

        StringBuilder[] rows = new StringBuilder[numRows];
        for (int i = 0; i < rows.length; i++)
            rows[i] = new StringBuilder();

        for (int i = 0; i < s.length(); i++) {
            int x = i % (2 * numRows - 2);
            if (x < numRows)
                rows[x].append(s.charAt(i));
            else
                rows[2 * numRows - 2 - x].append(s.charAt(i));
        }

        StringBuilder res = new StringBuilder();
        for (StringBuilder sb : rows)
            res.append(sb);
        return res.toString();
    }

    // Q7
    public int reverse(int x) {
        int result = 0;

        while (x != 0) {
            if (result < Integer.MIN_VALUE / 10 || result > Integer.MAX_VALUE / 10)
                return 0;
            int remainder = x % 10;
            x /= 10;
            result = result * 10 + remainder;
        }
        return result;
    }

    // Q8
    public int myAtoi(String s) {
        int sign = 1, ans = 0, index = 0;
        char[] array = s.toCharArray();

        while (index < array.length && array[index] == ' ')
            index++;

        if (index < array.length && (array[index] == '+' || array[index] == '-'))
            sign = array[index++] == '-' ? -1 : 1;

        while (index < array.length && array[index] <= '9' && array[index] >= '0') {
            int digit = array[index++] - '0';
            if (ans > (Integer.MAX_VALUE - digit) / 10)
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            ans = ans * 10 + digit;
        }
        return ans * sign;
    }

    // Q9
    public boolean isPalindrome(int x) {
        if (x < 0 || (x != 0 && x % 10 == 0))
            return false;

        int inversion = 0;
        while (x > inversion) {
            inversion = inversion * 10 + x % 10;
            x /= 10;
        }
        return x == inversion || x == inversion / 10;
    }

    // Q11
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1, ans = 0;

        while (left < right) {
            ans = height[left] < height[right] ? Math.max(ans, (right - left) * height[left++])
                    : Math.max(ans, (right - left) * height[right--]);
        }
        return ans;
    }

    // Q12
    public String intToRoman(int num) {
        int[] values = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
        String[] symbols = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };
        StringBuilder ans = new StringBuilder();

        for (int i = 0; i < values.length; i++) {
            while (num >= values[i]) {
                ans.append(symbols[i]);
                num -= values[i];

                if (num == 0)
                    break;
            }
        }
        return ans.toString();
    }

    // Q13
    public int romanToInt(String s) {
        Map<Character, Integer> map = new HashMap<Character, Integer>() {
            {
                put('I', 1);
                put('V', 5);
                put('X', 10);
                put('L', 50);
                put('C', 100);
                put('D', 500);
                put('M', 1000);
            }
        };
        int ans = 0;

        for (int i = 0; i < s.length(); i++) {
            if (i < s.length() - 1 && map.get(s.charAt(i)) < map.get(s.charAt(i + 1)))
                ans -= map.get(s.charAt(i));
            else
                ans += map.get(s.charAt(i));
        }
        return ans;
    }

    // Q14
    public String longestCommonPrefix(String[] strs) {
        for (int i = 0; i < strs[0].length(); i++) {
            char c = strs[0].charAt(i);
            for (int j = 1; j < strs.length; j++) {
                if (i == strs[j].length() || strs[j].charAt(i) != c)
                    return strs[0].substring(0, i);
            }
        }
        return strs[0];
    }

    public static void main(String[] args) {
        // Solution test = new Solution();

    }
}
