from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """Q1 alternative"""
        hashmap = {}
        for index, item in enumerate(nums):
            if target - item in hashmap.keys():
                return [index, hashmap[target - item]]
            hashmap[item] = index

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """Q1"""
        for i in range(len(nums)):
            if target - nums[i] in nums[i + 1:]:
                return [i, nums.index(target - nums[i], i + 1)]

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """Q2"""
        dummy = ListNode()
        cursor = dummy
        carry_flag = 0

        while l1 or l2 or carry_flag:
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0

            cursor.next = ListNode((x + y + carry_flag) % 10)
            carry_flag = (x + y + carry_flag) // 10

            cursor = cursor.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        return dummy.next

    def lengthOfLongestSubstring(self, s: str) -> int:
        """Q3"""
        n = len(s)
        right = 0
        window = set()
        result = 0

        for left in range(n):
            while right < n and s[right] not in window:
                window.add(s[right])
                result = max(result, len(window))
                right += 1
            window.remove(s[left])
        return result
