import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

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

    // Q15
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                return ans;
            }
            if (i != 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            int j = i + 1, k = nums.length - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == 0) {
                    ans.add(new ArrayList<>(Arrays.asList(nums[i], nums[j++], nums[k--])));

                    while (j < k && nums[j] == nums[j - 1])
                        j++;
                    while (j < k && nums[k] == nums[k + 1])
                        k--;
                } else if (sum < 0) {
                    j++;
                } else {
                    k--;
                }
            }
        }
        return ans;
    }

    // Q16
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int ans = nums[0] + nums[1] + nums[2];

        for (int i = 0; i < nums.length - 2; i++) {
            // skip same first num
            if (i > 0 && nums[i] == nums[i - 1])
                continue;

            int j = i + 1, k = nums.length - 1;
            int min = nums[i] + nums[j] + nums[j + 1];
            int max = nums[i] + nums[k - 1] + nums[k];

            // skip the rest loop as min increases
            if (target <= min) {
                if (Math.abs(target - min) < Math.abs(target - ans))
                    ans = min;
                break;
            }

            if (max <= target) {
                if (Math.abs(target - max) < Math.abs(target - ans))
                    ans = max;
                if (ans == target)
                    return ans;
                continue;
            }

            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == target)
                    return sum;
                if (Math.abs(target - sum) < Math.abs(target - ans))
                    ans = sum;
                if (sum > target) {
                    k--;
                    while (j < k && nums[k] == nums[k + 1])
                        k--;
                } else {
                    j++;
                    while (j < k && nums[j] == nums[j - 1])
                        j++;
                }
            }
        }
        return ans;
    }

    // Q17
    public List<String> letterCombinations(String digits) {
        List<String> ans = new ArrayList<>();
        if (digits.length() == 0)
            return ans;
        Map<Character, String> map = new HashMap<Character, String>() {
            {
                put('2', "abc");
                put('3', "def");
                put('4', "ghi");
                put('5', "jkl");
                put('6', "mno");
                put('7', "pqrs");
                put('8', "tuv");
                put('9', "wxyz");
            }
        };
        StringBuilder combination = new StringBuilder();
        letterCombinations_helper(digits, combination, ans, map);
        return ans;
    }

    void letterCombinations_helper(String currentDigits, StringBuilder currentCombination, List<String> ans,
            Map<Character, String> map) {
        if (currentDigits.length() == 0)
            ans.add(currentCombination.toString());
        else {
            String firstLetters = map.get(currentDigits.charAt(0));
            for (int i = 0; i < firstLetters.length(); i++) {
                char letter = firstLetters.charAt(i);
                letterCombinations_helper(currentDigits.substring(1), currentCombination.append(letter), ans, map);
                // need to delete the last char added, because of the modified StringBuilder.
                currentCombination.deleteCharAt(currentCombination.length() - 1);
            }
        }
    }

    // Q18
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        if (nums.length < 4)
            return ans;
        Arrays.sort(nums);

        int n = nums.length;
        for (int i = 0; i < n - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            if ((long) nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target)
                break;
            if ((long) nums[i] + nums[n - 3] + nums[n - 2] + nums[n - 1] < target)
                continue;

            for (int j = i + 1; j < n - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1])
                    continue;
                if ((long) nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target)
                    break;
                if ((long) nums[i] + nums[j] + nums[n - 2] + nums[n - 1] < target)
                    continue;

                int k = j + 1, l = n - 1;
                while (k < l) {
                    long sum = (long) nums[i] + nums[j] + nums[k] + nums[l];
                    if (sum == target) {
                        ans.add(Arrays.asList(nums[i], nums[j], nums[k++], nums[l--]));
                        while (k < l && nums[k] == nums[k - 1])
                            k++;
                        while (k < l && nums[l] == nums[l + 1])
                            l--;
                    } else if (sum < target) {
                        k++;
                    } else
                        l--;
                }
            }
        }
        return ans;
    }

    // Q19
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode fast = dummy, slow = dummy;
        for (int i = 0; i < n; i++)
            fast = fast.next;

        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }

    // Q20
    public boolean isValid(String s) {
        if (s.length() % 2 == 1)
            return false;

        Map<Character, Character> pairs = new HashMap<Character, Character>() {
            {
                put(')', '(');
                put(']', '[');
                put('}', '{');
            }
        };

        Deque<Character> stack = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            Character c = s.charAt(i);
            if (pairs.containsKey(c)) {
                if (stack.isEmpty() || stack.peek() != pairs.get(c))
                    return false;
                else
                    stack.pop();
            } else
                stack.push(c);
        }
        return stack.isEmpty();
    }

    // Q21
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        } else if (list2 == null) {
            return list1;
        } else if (list1.val < list2.val) {
            list1.next = this.mergeTwoLists(list1.next, list2);
            return list1;
        } else {
            list2.next = this.mergeTwoLists(list1, list2.next);
            return list2;
        }
    }

    // Q22
    public List<String> generateParenthesis(int n) {
        List<String> ans = new ArrayList<>();
        this.generateParenthesis_helper(ans, n, 0, 0, new StringBuilder());
        return ans;
    }

    void generateParenthesis_helper(List<String> ans, int n, int left, int right, StringBuilder current) {
        if (left == n && right == n) {
            ans.add(current.toString());
            return;
        }
        if (left < n) {
            current.append("(");
            this.generateParenthesis_helper(ans, n, left + 1, right, current);
            current.deleteCharAt(current.length() - 1);
        }
        if (right < left) {
            current.append(")");
            this.generateParenthesis_helper(ans, n, left, right + 1, current);
            current.deleteCharAt(current.length() - 1);
        }
    }

    // Q23
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val - o2.val; // >0, true, exchange
            }
        });

        ListNode dummy = new ListNode(), cursor = dummy;
        for (ListNode list : lists) {
            if (list != null)
                pq.add(list);
        }

        while (!pq.isEmpty()) {
            ListNode cursorrentSmall = pq.poll();
            cursor.next = cursorrentSmall;
            cursor = cursor.next;
            if (cursorrentSmall.next != null)
                pq.add(cursorrentSmall.next);
        }
        return dummy.next;
    }

    // Q24
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null)
            return head;

        ListNode newHead = head.next;
        head.next = this.swapPairs(newHead.next);
        newHead.next = head;
        return newHead;
    }

    // Q25
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode cursor = head;
        for (int i = 0; i < k; i++) {
            if (cursor == null)
                return head;
            cursor = cursor.next;
        }

        cursor = this.reverseKGroup(cursor, k);
        for (int i = 0; i < k; i++) {
            ListNode temp = head.next;
            head.next = cursor;
            cursor = head;
            head = temp;
        }
        return cursor;
    }

    // Q26
    public int removeDuplicates(int[] nums) {
        if (nums.length < 2)
            return nums.length;

        int slow = 1;
        for (int fast = slow; fast < nums.length; fast++) {
            if (nums[fast] != nums[fast - 1])
                nums[slow++] = nums[fast];
        }
        return slow;
    }

    // Q27
    public int removeElement(int[] nums, int val) {
        int left = 0, right = nums.length;
        while (left < right) {
            if (nums[left] == val)
                nums[left] = nums[--right];
            else
                left++;
        }
        return left;
    }

    // Q31
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1])
            i--;

        if (i >= 0) {
            int j = nums.length - 1;
            while (i < j && nums[i] >= nums[j])
                j--;
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }

        int left = i + 1, right = nums.length - 1;
        while (left < right) {
            int temp = nums[left];
            nums[left++] = nums[right];
            nums[right--] = temp;
        }
    }

    // Q32
    public int longestValidParentheses(String s) {
        int left = 0, right = 0, ans = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(')
                left++;
            else
                right++;
            if (left == right)
                ans = Math.max(ans, left + right);
            else if (left < right)
                left = right = 0;
        }

        left = right = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            if (s.charAt(i) == '(')
                left++;
            else
                right++;
            if (left == right)
                ans = Math.max(ans, left + right);
            else if (left > right)
                left = right = 0;
        }
        return ans;
    }

    // Q33
    public int search(int[] nums, int target) {
        int l = -1, r = nums.length;
        while (l + 1 < r) {
            int m = (l + r) / 2;
            if (nums[m] == target)
                return m;
            if (nums[l + 1] <= nums[m]) {
                if (nums[l + 1] <= target && target <= nums[m])
                    r = m;
                else
                    l = m;
            } else {
                if (nums[m] <= target && target <= nums[r - 1])
                    l = m;
                else
                    r = m;
            }
        }
        return -1;
    }

    // Q34
    public int[] searchRange(int[] nums, int target) {
        int l = -1, r = nums.length;
        if (r == 0)
            return new int[] { -1, -1 };

        while (l + 1 < r) {
            int m = (l + r) / 2;
            if (nums[m] < target)
                l = m;
            else
                r = m;
        }

        if (r == nums.length || nums[r] != target)
            return new int[] { -1, -1 };
        int[] ans = new int[] { r, -1 };

        l = -1;
        r = nums.length;
        while (l + 1 < r) {
            int m = (l + r) / 2;
            if (nums[m] <= target)
                l = m;
            else
                r = m;
        }
        ans[1] = l;
        return ans;
    }

    // Q35
    public int searchInsert(int[] nums, int target) {
        int l = -1, r = nums.length;
        while (l + 1 < r) {
            int m = (l + r) / 2;
            if (nums[m] < target)
                l = m;
            else
                r = m;
        }
        return r;
    }

    // Q36
    public boolean isValidSudoku(char[][] board) {
        boolean[][] row = new boolean[9][9];
        boolean[][] column = new boolean[9][9];
        boolean[][] box = new boolean[9][9];

        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.')
                    continue;
                int num = board[i][j] - '1';
                if (row[i][num] || column[j][num] || box[i / 3 * 3 + j / 3][num])
                    return false;
                row[i][num] = column[j][num] = box[i / 3 * 3 + j / 3][num] = true;
            }
        return true;
    }

    public static void main(String[] args) {
        // Solution test = new Solution();

    }
}
