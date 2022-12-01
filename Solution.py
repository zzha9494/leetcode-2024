from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """Q1"""
        hashmap = {}
        for index, item in enumerate(nums):
            if target - item in hashmap.keys():
                return [index, hashmap[target - item]]
            hashmap[item] = index

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """Q1 alternative"""
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

    def longestPalindrome(self, s: str) -> str:
        """Q5"""
        n = len(s)
        start = 0
        max_length = 1
        dp = [[False] * n for _ in range(n)]

        # Single char palindrome
        for i in range(n):
            dp[i][i] = True

        # Palindrome length between 2 to n
        for L in range(2, n+1):
            for left in range(n):
                # right - left + 1 = L
                right = L + left - 1

                if right >= n:
                    break

                if s[left] == s[right]:
                    if right - left == 1:
                        dp[left][right] = True
                    else:
                        dp[left][right] = dp[left + 1][right - 1]

                if dp[left][right] and L > max_length:
                    max_length = L
                    start = left

        return s[start: start + max_length]

    def convert(self, s: str, numRows: int) -> str:
        """Q6"""
        if numRows == 1:
            return s

        rows = ["" for _ in range(numRows)]

        for i in range(len(s)):
            # T = 2 * numRows - 2
            x = i % (2 * numRows - 2)
            if x < numRows:
                rows[x] += s[i]
            else:
                rows[2 * numRows - 2 - x] += s[i]
        return "".join(rows)

    def reverse(self, x: int) -> int:
        """Q7"""
        sign = -1 if x < 0 else 1
        x = sign * int(str(sign * x)[::-1])
        return x if x in range(-2 ** 31, 2 ** 31 - 1) else 0

    def myAtoi(self, s: str) -> int:
        """Q8"""
        sign, index, ans, n = 1, 0, 0, len(s)
        MIN_VAL, MAX_VAL = -2 ** 31, 2**31 - 1

        # s = s.lstrip()
        while index < n and s[index] == " ":
            index += 1

        if index < n and s[index] in "+-":
            sign = 1 if s[index] == "+" else -1
            index += 1

        while index < n and s[index].isdecimal():
            if ans > (MAX_VAL - int(s[index])) / 10:
                return MAX_VAL if sign == 1 else MIN_VAL
            ans = ans * 10 + int(s[index])
            index += 1
        return sign * ans

    def myAtoi(self, s: str) -> int:
        """Q8 alternative"""
        s = s.lstrip()
        if not s:
            return 0

        ans, sign = "", 1
        MIN_VAL, MAX_VAL = -2 ** 31, 2**31 - 1
        if s[0] in "+-":
            sign = 1 if s[0] == "+" else -1
            s = s[1:]

        for i in s:
            if i.isdecimal():
                ans += i
            else:
                break

        try:
            ans = sign * int(ans)
        except:
            return 0

        if ans in range(MIN_VAL, MAX_VAL):
            return ans
        else:
            return MIN_VAL if sign == -1 else MAX_VAL

    def isPalindrome(self, x: int) -> bool:
        """Q9"""
        return False if x < 0 else str(x) == str(x)[::-1]
