import java.util.HashMap;

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

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        // Q2
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
}
