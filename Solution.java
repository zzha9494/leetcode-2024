import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
// import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
// import java.util.Set;

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

    // Q38
    public String countAndSay(int n) {
        if (n == 1)
            return "1";

        String s = this.countAndSay(n - 1);
        int count = 0;
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            count++;
            if (i == s.length() - 1 || s.charAt(i) != s.charAt(i + 1)) {
                ans.append(count).append(s.charAt(i));
                count = 0;
            }
        }
        return ans.toString();
    }

    // Q39
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        this.combinationSum_helper(candidates, target, new LinkedList<>(), ans);
        return ans;
    }

    void combinationSum_helper(int[] candidates, int target, Deque<Integer> path, List<List<Integer>> ans) {
        for (int i = 0; i < candidates.length; i++) {
            if (!path.isEmpty() && candidates[i] < path.peek())
                continue;
            if (candidates[i] < target) {
                path.push(candidates[i]);
                this.combinationSum_helper(candidates, target - candidates[i], path, ans);
                path.pop();
            } else if (candidates[i] == target) {
                path.push(candidates[i]);
                ans.add(new ArrayList<>(path));
                path.pop();
                return;
            } else
                return;
        }
    }

    // Q40
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        this.combinationSum2_helper(candidates, target, new LinkedList<>(), ans);
        return ans;
    }

    void combinationSum2_helper(int[] candidates, int target, Deque<Integer> path, List<List<Integer>> ans) {
        for (int i = 0; i < candidates.length; i++) {
            if (i > 0 && candidates[i] == candidates[i - 1])
                continue;
            if (candidates[i] < target) {
                path.addFirst(candidates[i]);
                this.combinationSum2_helper(Arrays.copyOfRange(candidates, i + 1, candidates.length),
                        target - candidates[i], path,
                        ans);
                path.removeFirst();
            } else if (candidates[i] == target) {
                path.addFirst(candidates[i]);
                ans.add(new ArrayList<>(path));
                path.removeFirst();
                return;
            } else
                return;
        }
    }

    // Q41
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            // if number correct, exchange another with position number - 1
            // prevent dead loop such as nums[0] = 1 = nums[1 - 1]
            while (nums[i] >= 1 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                int temp = nums[i];
                nums[i] = nums[temp - 1];
                nums[temp - 1] = temp;
            }
        }

        for (int i = 0; i < n; i++) {
            // a correct number should equal to index + 1, such as nums[0] = 1, nums[1] = 2
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    // Q42
    public int trap(int[] height) {
        int ans = 0;
        int left = 0, right = height.length - 1;
        int leftMax = 0, rightMax = 0;
        while (left < right) {
            leftMax = Math.max(leftMax, height[left]);
            rightMax = Math.max(rightMax, height[right]);
            if (leftMax < rightMax) {
                ans += leftMax - height[left++];
            } else {
                ans += rightMax - height[right--];
            }
        }
        return ans;
    }

    // Q45
    public int jump(int[] nums) {
        int step = 0, maxPosition = 0, border = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            maxPosition = Math.max(maxPosition, i + nums[i]);
            if (i == border) {
                step++;
                border = maxPosition;
                if (border >= nums.length - 1) {
                    break;
                }
            }
        }
        return step;
    }

    // Q46
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();

        List<Integer> output = new ArrayList<Integer>();
        for (int num : nums) {
            output.add(num);
        }

        int n = nums.length;
        permute_helper(0, n, output, ans);
        return ans;
    }

    void permute_helper(int first, int n, List<Integer> output, List<List<Integer>> ans) {
        if (first == n) {
            ans.add(new ArrayList<>(output));
        }
        for (int i = first; i < n; i++) {
            Collections.swap(output, first, i);
            permute_helper(first + 1, n, output, ans);
            Collections.swap(output, first, i);
        }
    }

    // Q47
    public List<List<Integer>> permuteUnique(int[] nums) {
        boolean[] used = new boolean[nums.length];
        List<Integer> track = new ArrayList<>();
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        permuteUnique_helper(nums, used, track, ans);
        return ans;
    }

    void permuteUnique_helper(int[] nums, boolean[] used, List<Integer> track, List<List<Integer>> ans) {
        if (track.size() == nums.length) {
            ans.add(new ArrayList<>(track));
            return;
        }
        // the level of the for loop represents the index where a number is assigned,
        // for example, the top level is to choose the first number for index 0.
        for (int i = 0; i < nums.length; i++) {
            if (used[i] || (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])) {
                continue;
            }
            used[i] = true;
            track.add(nums[i]);
            this.permuteUnique_helper(nums, used, track, ans);
            track.remove(track.size() - 1);
            used[i] = false;
        }
    }

    // Q48
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - i][j];
                matrix[n - 1 - i][j] = temp;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }

    // Q49
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);

            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(str);

            map.put(key, list);
        }
        return new ArrayList<>(map.values());
    }

    // Q50
    public double myPow(double x, int n) {
        long N = n;
        return N >= 0 ? myPow_helper(x, N) : 1.0 / myPow_helper(x, -N);
    }

    double myPow_helper(double x, long N) {
        double ans = 1;
        double x_contribute = x;

        // 5 = (101)_2 = 1 * x^1 * x^4
        while (N > 0) {
            // last digit is 1, multiply
            if (N % 2 == 1) {
                ans *= x_contribute;
            }
            x_contribute *= x_contribute;
            N /= 2;

        }
        return ans;
    }

    // Q51
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> ans = new ArrayList<>();
        List<String> board = new ArrayList<>();
        boolean[] columns = new boolean[n];
        boolean[] diagonal_1 = new boolean[2 * n - 1];
        boolean[] diagonal_2 = new boolean[2 * n - 1];
        boolean[][] records = { columns, diagonal_1, diagonal_2 };
        solveNQueens_helper(0, n, records, board, ans);
        return ans;
    }

    void solveNQueens_helper(int row, int n, boolean[][] records, List<String> board, List<List<String>> ans) {
        if (row == n) {
            ans.add(new ArrayList<>(board));
            return;
        }

        char[] current = new char[n]; // optimize, current row to be added
        Arrays.fill(current, '.'); // optimize
        for (int column = 0; column < n; column++) {
            if (records[0][column] || records[1][n - 1 + column - row] || records[2][column + row]) {
                continue;
            }
            current[column] = 'Q'; // optimize
            board.add(String.copyValueOf(current)); // optimize
            records[0][column] = true;
            records[1][n - 1 + column - row] = true;
            records[2][+column + row] = true;
            this.solveNQueens_helper(row + 1, n, records, board, ans);
            current[column] = '.';
            board.remove(row);
            records[0][column] = false;
            records[1][n - 1 + column - row] = false;
            records[2][column + row] = false;
        }
    }

    // Q52
    public int totalNQueens(int n) {
        boolean[][] records = { new boolean[n], new boolean[2 * n - 1], new boolean[2 * n - 1] };
        return this.totalNQueens_helper(0, n, records);
    }

    int totalNQueens_helper(int row, int n, boolean[][] records) {
        int ans = 0;
        if (row == n) {
            ans++;
        }
        for (int column = 0; column < n; column++) {
            if (records[0][column] || records[1][n - 1 + column - row] || records[2][column + row]) {
                continue;
            }
            records[0][column] = true;
            records[1][n - 1 + column - row] = true;
            records[2][+column + row] = true;
            ans += this.totalNQueens_helper(row + 1, n, records);
            records[0][column] = false;
            records[1][n - 1 + column - row] = false;
            records[2][column + row] = false;
        }
        return ans;
    }

    // Q53
    public int maxSubArray(int[] nums) {
        int ans = nums[0], pre = 0;
        for (int x : nums) {
            pre = Math.max(x, pre + x);
            ans = Math.max(ans, pre);
        }
        return ans;
    }

    // Q54
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> ans = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return ans;
        }
        int rows = matrix.length, columns = matrix[0].length;
        int left = 0, right = columns - 1, top = 0, bottom = rows - 1;
        while (true) {
            for (int j = left; j <= right; j++) {
                ans.add(matrix[top][j]);
            }
            if (++top > bottom)
                break;
            for (int i = top; i <= bottom; i++) {
                ans.add(matrix[i][right]);
            }
            if (left > --right)
                break;
            for (int j = right; j >= left; j--) {
                ans.add(matrix[bottom][j]);
            }
            if (top > --bottom)
                break;
            for (int i = bottom; i >= top; i--) {
                ans.add(matrix[i][left]);
            }
            if (++left > right)
                break;
        }
        return ans;
    }

    // Q55
    public boolean canJump(int[] nums) {
        int maxPosition = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > maxPosition)
                break;
            maxPosition = Math.max(maxPosition, i + nums[i]);
            if (maxPosition >= nums.length - 1)
                return true;
        }
        return false;
    }

    // Q56
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare(int[] i1, int[] i2) {
                return i1[0] - i2[0];
            }
        });

        Deque<int[]> ans = new LinkedList<>();
        for (int[] interval : intervals) {
            if (ans.isEmpty() || ans.peek()[1] < interval[0]) {
                ans.push(interval);
            } else {
                ans.peek()[1] = Math.max(ans.peek()[1], interval[1]);
            }
        }
        return ans.toArray(new int[ans.size()][]);
    }

    // ------------------------------------------------------------------------

    public static void main(String[] args) {
        // Solution s = new Solution();

    }
}
