import collections
import itertools
import math
from functools import lru_cache
from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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

    def maxArea(self, height: List[int]) -> int:
        """Q11"""
        left, right, ans = 0, len(height) - 1, 0
        max_height = max(height)

        while left < right:
            ans = max(ans, (right - left) * min(height[left], height[right]))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
            if ans >= max_height * (right - left):
                break
        return ans

    def intToRoman(self, num: int) -> str:
        """Q12"""
        VALUE_SYMBOLS = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ]
        ans = ""
        for value, symbol in VALUE_SYMBOLS:
            while num >= value:
                ans += symbol
                num -= value
                if num == 0:
                    break
        return ans

    def romanToInt(self, s: str) -> int:
        """Q13"""
        symbols = ["M", "CM", "D", "CD", "C", "XC",
                   "L", "XL", "X", "IX", "V", "IV", "I"]
        nums = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        ans, i = 0, 0

        while i < len(s):
            if i < len(s) - 1 and s[i:i+2] in symbols:
                ans += nums[symbols.index(s[i:i+2])]
                i += 2
            else:
                ans += nums[symbols.index(s[i])]
                i += 1
        return ans

    def longestCommonPrefix(self, strs: List[str]) -> str:
        """Q14"""
        for i in range(len(strs[0])):
            # for str in strs[1:]:
            #     if i == len(str) or str[i] != strs[0][i]:
            #         return str[:i]
            if any(i == len(str) or str[i] != strs[0][i] for str in strs[1:]):
                return strs[0][:i]
        return strs[0]

    def longestCommonPrefix(self, strs: List[str]) -> str:
        """Q14 alternative"""
        box = set()
        index = 0
        try:
            while True:
                for str in strs:
                    box.add(str[index])
                if len(box) != 1:
                    break
                box.clear()
                index += 1
        except IndexError:
            pass
        return strs[0][:index]

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """Q15"""
        n, ans = len(nums), []
        nums.sort()

        for i in range(n):
            if nums[i] > 0:
                return ans
            if i != 0 and nums[i] == nums[i-1]:
                continue

            j, k = i + 1, n - 1
            while j < k:
                sum = nums[i] + nums[j] + nums[k]
                if sum == 0:
                    ans.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1
                    while j < k and nums[j] == nums[j+1]:
                        j += 1
                    while j < k and nums[k] == nums[k-1]:
                        k -= 1
                elif sum < 0:
                    j += 1
                else:
                    k -= 1
        return ans

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        """Q16"""
        nums.sort()
        n = len(nums)
        ans = sum(nums[:3])

        for i in range(n-2):
            if i > 0 and nums[i] == nums[i-1]:
                continue

            j, k = i + 1, n - 1
            current_max = nums[i] + nums[k - 1] + nums[k]
            current_min = nums[i] + nums[j] + nums[j + 1]

            if current_min >= target:
                if abs(target - current_min) < abs(target - ans):
                    ans = current_min
                break  # current_min becomes larger as i increases

            if current_max <= target:
                if abs(target - current_max) < abs(target - ans):
                    ans = current_max
                    if ans == target:
                        return ans
                continue

            while j < k:
                temp = nums[i] + nums[j] + nums[k]
                if temp == target:
                    return temp
                elif abs(target - temp) < abs(target - ans):
                    ans = temp

                if temp > target:
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
                else:
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
        return ans

    def letterCombinations(self, digits: str) -> List[str]:
        """Q17"""
        if digits == "":
            return []

        dic = {'2': 'abc',
               '3': 'def',
               '4': 'ghi',
               '5': 'jkl',
               '6': 'mno',
               '7': 'pqrs',
               '8': 'tuv',
               '9': 'wxyz'}

        def letterCombinations_helper(current_digits, current_combination):
            if len(current_digits) == 0:
                ans.append(current_combination)
            else:
                for letter in dic[current_digits[0]]:
                    letterCombinations_helper(
                        current_digits[1:], current_combination + letter)

        ans = []
        letterCombinations_helper(digits, "")
        return ans

    def letterCombinations(self, digits: str) -> List[str]:
        """Q17 alternative"""
        if digits == "":
            return []

        dic = {'2': 'abc',
               '3': 'def',
               '4': 'ghi',
               '5': 'jkl',
               '6': 'mno',
               '7': 'pqrs',
               '8': 'tuv',
               '9': 'wxyz'}
        groups = [dic[digit] for digit in digits]
        return ["".join(combination) for combination in itertools.product(*groups)]

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """Q18"""
        ans = list()
        n = len(nums)
        if not nums or n < 4:
            return ans

        nums.sort()
        for i in range(n-3):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            if nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target:
                # current min
                break
            if nums[i] + nums[n-3] + nums[n-2] + nums[n-1] < target:
                # current max
                continue

            for j in range(i+1, n-2):
                if j > i+1 and nums[j] == nums[j-1]:
                    continue
                if nums[i] + nums[j] + nums[j+1] + nums[j+2] > target:
                    break
                if nums[i] + nums[j] + nums[n-2] + nums[n-1] < target:
                    continue

                k, l = j + 1, n - 1
                while k < l:
                    temp = nums[i] + nums[j] + nums[k] + nums[l]
                    if temp == target:
                        ans.append([nums[i], nums[j], nums[k], nums[l]])
                        k += 1
                        l -= 1
                        while k < l and nums[k] == nums[k-1]:
                            k += 1
                        while k < l and nums[l] == nums[l+1]:
                            l -= 1
                    elif temp < target:
                        k += 1
                    else:
                        l -= 1
        return ans

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """Q19"""
        dummy = ListNode(0, head)

        fast, slow = dummy, dummy
        for i in range(n):
            fast = fast.next

        while fast.next:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next
        return dummy.next

    def isValid(self, s: str) -> bool:
        """Q20"""
        if len(s) % 2 == 1:
            return False

        pairs = {
            ")": "(",
            "]": "[",
            "}": "{",
        }
        stack = list()
        for i in s:
            if i in pairs:
                if not stack or stack[-1] != pairs[i]:
                    return False
                stack.pop()
            else:
                stack.append(i)
        return not stack

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """Q21"""
        dummy = ListNode()
        cursor = dummy

        while list1 and list2:
            if list1.val < list2.val:
                cursor.next = list1
                list1 = list1.next
                cursor = cursor.next
            else:
                cursor.next = list2
                list2 = list2.next
                cursor = cursor.next
        if list1:
            cursor.next = list1
        else:
            cursor.next = list2
        return dummy.next

    @lru_cache(None)
    def generateParenthesis(self, n: int) -> List[str]:
        """Q22"""
        if n == 0:
            return [""]

        ans = []
        for i in range(n):
            for left in self.generateParenthesis(i):
                for right in self.generateParenthesis(n-i-1):
                    # i + n - i - 1 + 1 = n
                    ans.append(f"({left}){right}")
        return ans

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """Q23"""
        def mergeKLists_helper(left: ListNode, right: ListNode):
            # merge two lists
            dummy = ListNode()
            cursor = dummy

            while left and right:
                if left.val < right.val:
                    cursor.next = left
                    left = left.next
                    cursor = cursor.next
                else:
                    cursor.next = right
                    right = right.next
                    cursor = cursor.next
            if left:
                cursor.next = left
            else:
                cursor.next = right
            return dummy.next

        n = len(lists)
        # base case
        if n == 0:
            return None
        elif n < 2:
            return lists[0]

        # divide
        mid = n // 2
        left = lists[:mid]
        right = lists[mid:]

        # recur
        merged_left = self.mergeKLists(left)
        merged_right = self.mergeKLists(right)

        # conquer
        return mergeKLists_helper(merged_left, merged_right)

    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """Q24"""
        dummy = ListNode(next=head)
        cursor = dummy

        while cursor.next and cursor.next.next:
            # cursor -> node1 -> node2
            node1 = cursor.next
            node2 = node1.next

            cursor.next = node2
            node1.next = node2.next
            node2.next = node1
            cursor = node1
        return dummy.next

    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """Q25"""
        cursor = head
        for _ in range(k):
            if not cursor:
                return head
            cursor = cursor.next

        cursor = self.reverseKGroup(cursor, k)
        for _ in range(k):
            temp = head.next
            head.next = cursor
            cursor = head
            head = temp
        return cursor

    def removeDuplicates(self, nums: List[int]) -> int:
        """Q26"""
        n = len(nums)
        if n < 2:
            return n

        slow = 1
        for fast in range(1, n):
            if nums[fast] != nums[fast - 1]:
                nums[slow] = nums[fast]
                slow += 1
        return slow

    def removeElement(self, nums: List[int], val: int) -> int:
        """Q27"""
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        return slow

    def removeElement(self, nums: List[int], val: int) -> int:
        """Q27 alternative"""
        left, right = 0, len(nums)
        while left < right:
            if nums[left] == val:
                nums[left] = nums[right - 1]
                right -= 1
            else:
                left += 1
        return left

    def nextPermutation(self, nums: List[int]) -> None:
        """Q31"""
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i+1]:
            i -= 1

        if i >= 0:
            j = len(nums) - 1
            while i < j and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]

        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    def longestValidParentheses(self, s: str) -> int:
        """Q32"""
        stack = [-1]  # stack[-1] maintain the last element that is not matched
        ans = 0
        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()
                if len(stack) == 0:
                    stack.append(i)
                else:
                    ans = max(ans, i - stack[-1])
        return ans

    def search(self, nums: List[int], target: int) -> int:
        """Q33"""
        # border
        if nums[0] == target:
            return 0
        if nums[len(nums)-1] == target:
            return len(nums) - 1

        l, r = 0, len(nums) - 1
        while l + 1 < r:
            m = (l + r)//2
            if nums[m] == target:
                return m
            if nums[l] < nums[m]:  # [l,m] is ascending
                if nums[l] <= target and target <= nums[m]:
                    r = m
                else:
                    l = m
            else:
                if nums[m] <= target and target <= nums[r]:
                    l = m
                else:
                    r = m
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """Q34"""
        l, r = -1, len(nums)
        if r == 0:
            return [-1, -1]

        ans = []
        while l+1 < r:
            m = (l + r)//2
            if nums[m] >= target:
                r = m
            else:
                l = m
        if r == len(nums) or nums[r] != target:
            return [-1, -1]
        else:
            ans.append(r)

        l, r = -1, len(nums)
        while l+1 < r:
            m = (l + r) // 2
            if nums[m] > target:
                r = m
            else:
                l = m

        ans.append(l)
        return ans

    def searchInsert(self, nums: List[int], target: int) -> int:
        """Q35"""
        l, r = -1, len(nums)
        while l + 1 < r:
            m = (l + r) // 2
            if nums[m] < target:
                l = m
            else:
                r = m
        return r

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        """Q36"""
        row = [
            [False] * 9 for i in range(9)]  # whether current number appeared in the i-th row
        column = [[False] * 9 for i in range(9)]
        box = [[False] * 9 for i in range(9)]

        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    continue
                num = ord(board[i][j]) - ord("1")
                if row[i][num] or column[j][num] or box[i//3 * 3 + j//3][num]:
                    return False
                row[i][num], column[j][num] = True, True
                box[i//3 * 3 + j//3][num] = True,
        return True

    def countAndSay(self, n: int) -> str:
        """Q38"""
        if n == 1:
            return "1"

        s = self.countAndSay(n-1)
        count, ans = 0, ""
        for i in range(len(s)):
            count += 1
            if i == len(s) - 1 or s[i] != s[i+1]:
                ans += str(count) + s[i]
                count = 0
        return ans

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """Q39"""
        def combinationSum_helper(candidates: List[int], target: int, path: List[int], ans: List[List[int]]):
            for i in candidates:
                if path and i < path[-1]:
                    continue
                if i < target:
                    path.append(i)
                    combinationSum_helper(candidates, target - i, path, ans)
                    path.pop()
                elif i == target:
                    path.append(i)
                    ans.append(path.copy())
                    path.pop()
                    return
                else:
                    return

        candidates.sort()
        ans = []
        combinationSum_helper(candidates, target, [], ans)
        return ans

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """Q40"""
        def combinationSum2_helper(candidates: List[int], target: int, path: List[int], ans: List[List[int]]):
            for i in range(len(candidates)):
                # skip the duplicated number, such as [1, 1, 2, 3] target = 3
                if i > 0 and candidates[i] == candidates[i-1]:
                    continue
                if candidates[i] < target:
                    path.append(candidates[i])
                    combinationSum2_helper(
                        candidates[i+1:], target - candidates[i], path, ans)
                    path.pop()
                elif candidates[i] == target:
                    path.append(candidates[i])
                    ans.append(path.copy())
                    path.pop()
                    return
                else:
                    return

        candidates.sort()
        ans = []
        combinationSum2_helper(candidates, target, [], ans)
        return ans

    def firstMissingPositive(self, nums: List[int]) -> int:
        """Q41"""
        n = len(nums)
        # first loop remove non-positive
        for i in range(n):
            if nums[i] < 1:
                nums[i] = n + 1

        # if number correct, label position i-1 to minus sign
        for i in range(n):
            x = abs(nums[i])
            if x <= n:
                nums[x - 1] = -abs(nums[x - 1])

        # find the number without a label (minus sign)
        for i in range(n):
            if nums[i] > 0:
                return i + 1
        return n + 1

    def trap(self, height: List[int]) -> int:
        """Q42"""
        n = len(height)
        highest = max(height)
        # invalid input
        if n <= 2 or highest == 0:
            return 0

        ans = 0
        left, current = 0, 0
        # from left to first highest
        while height[left] != highest:
            if height[left] <= current:
                # if left is lower than current border, can get some water
                ans += (current - height[left])
            else:
                # update the higher border
                current = height[left]
            left += 1

        right, current = n - 1, 0
        # from right to last highest
        while height[right] != highest:
            if height[right] <= current:
                ans += (current - height[right])
            else:
                current = height[right]
            right -= 1

        # if it has more than one highest
        while left != right:
            ans += (highest - height[left])
            left += 1

        return ans

    def jump(self, nums: List[int]) -> int:
        """Q45"""
        maxPosition = 0  # the farthest position where next jump can reach
        border = 0  # the farthest position where the current jump can reach
        step = 0
        for i in range(len(nums) - 1):
            # within the current jump, find the farthest position for next jump
            maxPosition = max(maxPosition, i + nums[i])
            # search until border, have to jump, no need to jump farthest
            if i == border:
                step += 1
                border = maxPosition
                if maxPosition >= len(nums) - 1:
                    break
        return step

    def permute(self, nums: List[int]) -> List[List[int]]:
        """Q46"""
        n = len(nums)
        if n == 1:
            return [nums]
        else:
            current = self.permute(nums[1:])
            ans = []
            for e in current:
                for i in range(len(e) + 1):
                    temp = e.copy()
                    temp.insert(i, nums[0])
                    ans.append(temp)
            return ans

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """Q47"""
        def permuteUnique_helper(start: int, n: int, output: List[int], ans: set[tuple[int]]):
            if start == n:
                ans.add(tuple(output))
                return
            for i in range(start, n):
                output[start], output[i] = output[i], output[start]
                permuteUnique_helper(start+1, n, output, ans)
                output[start], output[i] = output[i], output[start]

        ans = set()
        permuteUnique_helper(0, len(nums), nums, ans)
        res = []
        for e in ans:
            res.append(list(e))
        return res

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """Q47 alternative"""
        def permuteUnique_helper(current: int, n: int, output: List[int], ans: List[List[int]]):
            if current == n:
                ans.append(output.copy())
                return
            used = set()
            for i in range(current, n):
                # [1, 1, 2], current = 0, i = 0
                # current means choose a number for current index
                # first loop, 1 have not used, swap with current, iterate next with[1, 1, 2]
                # current = 0, i = 1
                # second loop, 1 have already used, if swap, still be [1, 1, 2], duplicated
                if output[i] in used:
                    continue
                used.add(output[i])
                output[current], output[i] = output[i], output[current]
                permuteUnique_helper(current+1, n, output, ans)
                output[current], output[i] = output[i], output[current]

        ans = []
        nums.sort()
        permuteUnique_helper(0, len(nums), nums, ans)
        return ans

    def rotate(self, matrix: List[List[int]]) -> None:
        """Q48"""
        n = len(matrix)
        # first, horizontal flip
        for row in range(n // 2):
            for column in range(n):
                matrix[row][column], matrix[n - 1-row][column] = \
                    matrix[n - 1 - row][column], matrix[row][column]

        # then, diagonal flip
        for row in range(n):
            for column in range(row, n):
                matrix[row][column], matrix[column][row] = \
                    matrix[column][row], matrix[row][column]

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """Q49"""
        mp = collections.defaultdict(list)

        for s in strs:
            key = "".join(sorted(s))
            mp[key].append(s)

        return list(mp.values())

    def myPow(self, x: float, n: int) -> float:
        """Q50"""
        # closure, helper can access outside variables such as x
        def myPow_helper(n: int) -> float:
            if n == 0:
                return 1
            half = myPow_helper(n // 2)
            return half * half if n % 2 == 0 else half * half * x
        return myPow_helper(n) if n >= 0 else 1.0 / myPow_helper(-n)

    def solveNQueens(self, n: int) -> List[List[str]]:
        """Q51"""
        def solveNQueens_helper(row):
            # new solution found
            if row == n:
                ans.append(board.copy())
                return

            # iterate every column in current row
            for column in range(n):
                # not a valid position, pass
                if columns[column] or diagonal_1[column-row] or diagonal_2[column + row]:
                    continue
                # choose the position for this row, set records and enter next row
                board.append("." * column + "Q" + "." * (n - 1 - column))
                columns[column], diagonal_1[column - row], diagonal_2[column + row] = \
                    True, True, True
                solveNQueens_helper(row + 1)
                # backtrack
                board.pop()
                columns[column], diagonal_1[column - row], diagonal_2[column + row] = \
                    False, False, False

        ans = []
        board = []
        columns = [False] * n
        diagonal_1 = [False] * (2 * n - 1)  # \: -i+j # 2(n-1) + 1
        diagonal_2 = [False] * (2 * n - 1)  # /: i+j

        # loop from first row
        solveNQueens_helper(0)
        return ans

    def totalNQueens(self, n: int) -> int:
        """Q52"""
        def totalNQueens_helper(row):
            if row == n:
                nonlocal ans
                ans += 1
            else:
                for column in range(n):
                    if columns[column] or diagonal_1[column-row] or diagonal_2[column+row]:
                        continue
                    columns[column], diagonal_1[column-row], diagonal_2[column+row] =\
                        True, True, True
                    totalNQueens_helper(row + 1)
                    columns[column], diagonal_1[column-row], diagonal_2[column+row] =\
                        False, False, False

        columns = [False] * n
        diagonal_1 = [False] * (2 * n - 1)  # \: -i+j # 2(n-1) + 1
        diagonal_2 = [False] * (2 * n - 1)  # /: i+j
        ans = 0
        totalNQueens_helper(0)
        return ans

    def maxSubArray(self, nums: List[int]) -> int:
        """Q53"""
        # dp[i] represents the max subarray sum until i (including)
        dp = [nums[0]]
        # dp[i] = nums[i] + dp[i-1], if dp[i-1] > 0 (positive contribution)
        # dp[i] = nums[i], if if dp[i-1] <= 0 (negative contribution)
        for i in range(1, len(nums)):
            dp.append(nums[i] if dp[-1] < 0 else dp[-1] + nums[i])
        return max(dp)

    def maxSubArray(self, nums: List[int]) -> int:
        """Q53 alternative"""
        # because dp[i] is only related to dp[i-1] and nums[i], can save space from o(n) to o(1)
        ans = nums[0]
        pre = 0  # when i, it means dp[i-1]
        for i in range(len(nums)):
            pre = max(pre + nums[i], nums[i])
            ans = max(ans, pre)
        return ans

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """Q54"""
        ans = []
        while matrix:
            ans += matrix.pop(0)
            # *matrix returns a iterator of each row
            # zip() returns an iterator of tuples, where the i-th tuple contains the i-th element from each of the argument iterables.
            matrix = list(zip(*matrix))[::-1]
        return ans

    def canJump(self, nums: List[int]) -> bool:
        """Q55"""
        maxPosition = 0
        for i in range(len(nums)):
            if i > maxPosition:
                break
            maxPosition = max(maxPosition, i + nums[i])
            if maxPosition >= len(nums) - 1:
                return True
        return False

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """Q56"""
        intervals.sort(key=lambda x: x[0])
        ans = []
        left, right = intervals[0]
        for e in intervals:
            if e[0] > right:
                ans.append([left, right])
                left, right = e
            else:
                right = max(right, e[1])
        ans.append([left, right])
        return ans

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """Q57"""
        intervals.append(newInterval)
        intervals.sort(key=lambda x: x[0])
        ans = []
        for e in intervals:
            if not ans or e[0] > ans[-1][1]:
                ans.append(e)
            else:
                ans[-1][1] = max(ans[-1][1], e[1])
        return ans

    def lengthOfLastWord(self, s: str) -> int:
        """Q58"""
        s = s.rstrip()
        ans = 0
        for c in s[::-1]:
            if c != " ":
                ans += 1
            else:
                break
        return ans

    def generateMatrix(self, n: int) -> List[List[int]]:
        """Q59"""
        matrix = [[None] * n for _ in range(n)]
        num = 1
        top, right, bottom, left = 0, n-1, n-1, 0
        while (top <= bottom and left <= right):
            for j in range(left, right + 1):
                matrix[top][j] = num
                num += 1
            for i in range(top + 1, bottom + 1):
                matrix[i][right] = num
                num += 1
            for j in range(right - 1, left - 1, -1):
                matrix[bottom][j] = num
                num += 1
            for i in range(bottom-1, top, -1):
                matrix[i][left] = num
                num += 1

            top += 1
            right -= 1
            bottom -= 1
            left += 1
        return matrix

    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """Q61"""
        dummy = ListNode(0, head)
        cursor = dummy
        n = 0
        while cursor.next:
            n += 1
            cursor = cursor.next

        if not n or not k or not k % n:
            return head

        k %= n
        fast, slow = dummy, dummy
        for i in range(k):
            fast = fast.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        fast.next = head
        head = slow.next
        slow.next = None
        return head

    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """Q61 alternative"""
        if k == 0 or not head or not head.next:
            return head

        n = 1
        cursor = head
        while cursor.next:
            cursor = cursor.next
            n += 1

        if (add := n-k % n) == n:
            return head

        cursor.next = head
        while add:
            cursor = cursor.next
            add -= 1

        newHead = cursor.next
        cursor.next = None
        return newHead

    def uniquePaths(self, m: int, n: int) -> int:
        """Q62"""
        dp = [[1] * n] + [[1] + [0] * (n-1) for _ in range(m-1)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """Q62 alternative"""
        n, m = min(m, n), max(m, n)  # reduce the space to O(min(m, n))
        dp = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j-1]
        return dp[n-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """Q62 alternative"""
        return math.comb(m + n - 2, n-1)  # choose n+1 from m-1 + n-1

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """Q63"""
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [0] * n
        dp[0] = 0 if obstacleGrid[0][0] else 1

        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j]:
                    dp[j] = 0
                    continue
                if j >= 1:
                    dp[j] += dp[j-1]
        return dp[n-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """Q64"""
        m, n = len(grid), len(grid[0])

        for j in range(1, n):
            grid[0][j] = grid[0][j-1] + grid[0][j]

        for i in range(1, m):
            for j in range(n):
                if j == 0:
                    grid[i][0] = grid[i-1][0] + grid[i][0]
                else:
                    grid[i][j] = grid[i][j] + min(grid[i-1][j], grid[i][j-1])
        return grid[m-1][n-1]

    def plusOne(self, digits: List[int]) -> List[int]:
        """Q66"""
        n = len(digits)
        for i in range(n):
            if digits[n-1-i] == 9:
                continue
            if i == 0:
                digits[-1] += 1
                return digits
            digits[n-1-i] += 1
            digits[-i:] = [0] * (i)  # 0 to i-1 is 9
            return digits
        return [1] + [0] * n

    def addBinary(self, a: str, b: str) -> str:
        """Q67"""
        return bin(int(a, 2) + int(b, 2))[2:]

    def mySqrt(self, x: int) -> int:
        """Q69"""
        l, r = -1, x+1
        while l + 1 < r:
            m = (l + r)//2
            if m * m <= x:
                l = m
            else:
                r = m
        return l

    def climbStairs(self, n: int) -> int:
        """Q70"""
        p, q, r = 0, 0, 1
        for i in range(n):
            p = q
            q = r
            r = p+q
        return r

    def simplifyPath(self, path: str) -> str:
        """Q71"""
        path = path.split("/")
        stack = []
        for e in path:
            if not e or e == "." or (e == ".." and not stack):
                continue
            if e == "..":
                stack.pop()
            else:
                stack.append(e)
        return "/" + "/".join(stack)

    def minDistance(self, word1: str, word2: str) -> int:
        """Q72"""
        # https://leetcode.cn/problems/edit-distance/solutions/188223/bian-ji-ju-chi-by-leetcode-solution/comments/331399
        m, n = len(word1), len(word2)
        dp = [i for i in range(n + 1)]  # 0 ... n
        for i in range(1, m + 1):
            leftUp = dp[0]  # keep dp[i-1][j-1]
            dp[0] = i
            for j in range(1, n + 1):
                dp[j], leftUp = min(
                    1 + dp[j],  # "abcde"->"abcd"->"fgh"
                    dp[j-1] + 1,  # "abcde"->"fg"->"fgh"
                    # "abcde"->"fge"->"fgh"
                    leftUp + int(word1[i-1] != word2[j-1])
                ), dp[j]
        return dp[-1]

    def setZeroes(self, matrix: List[List[int]]) -> None:
        """Q73"""
        m, n = len(matrix), len(matrix[0])
        flag_column = any(matrix[i][0] == 0 for i in range(m))
        flag_row = any(matrix[0][j] == 0 for j in range(n))

        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0

        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        if flag_column:
            for i in range(m):
                matrix[i][0] = 0

        if flag_row:
            for j in range(n):
                matrix[0][j] = 0

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """Q74"""
        m, n = len(matrix), len(matrix[0])
        l, r = -1, m*n
        while l+1 != r:
            mid = (l+r) // 2
            i, j = mid//n, mid % n
            if matrix[i][j] == target:
                return True
            if matrix[i][j] < target:
                l = mid
            else:
                r = mid
        return False

    def sortColors(self, nums: List[int]) -> None:
        """Q75"""
        pointer_zero, pointer_two = 0, len(nums) - 1
        i = 0
        while i <= pointer_two:
            # prevent index out of range
            while i <= pointer_two and nums[i] == 2:
                nums[i], nums[pointer_two] = nums[pointer_two], nums[i]
                pointer_two -= 1
            if nums[i] == 0:
                nums[i], nums[pointer_zero] = nums[pointer_zero], nums[i]
                pointer_zero += 1
            i += 1

    def minWindow(self, s: str, t: str) -> str:
        """Q76"""
        need = collections.Counter(t)
        ans = [0, float('inf')]
        counter = len(t)
        left = 0
        for right, char in enumerate(s):
            if need[char] > 0:
                counter -= 1
            # every char will be deducted by 1
            need[char] -= 1
            if counter == 0:
                while True:
                    c = s[left]
                    if need[c] == 0:
                        # for char not needed, right minus A times, left plus B time
                        # A>=B, thus not break when char not needed
                        break
                    need[c] += 1
                    left += 1

                if right - left < ans[1] - ans[0]:
                    ans = [left, right]

                # remove a char needed
                need[s[left]] += 1
                counter += 1
                left += 1

        return s[ans[0]: ans[1] + 1] if ans[1] != float('inf') else ""

    def combine(self, n: int, k: int) -> List[List[int]]:
        """Q77"""
        return list(itertools.combinations(range(1, n+1), k))

    def combine(self, n: int, k: int) -> List[List[int]]:
        """Q77 alternative"""
        ans = []
        if k == 1:
            return [[i] for i in range(1, n+1)]
        if k == n:
            return [list(range(1, n+1))]
        # choosing 2 from [1,2,3,4] equals to
        # choosing 1 from [1,2,3] + [4] plus choosing 2 from [1,2,3]
        return [(i + [n]) for i in self.combine(n-1, k-1)] + self.combine(n-1, k)

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """Q78"""
        ans = []
        if len(nums) == 0:
            ans.append([])
        else:
            temp = self.subsets(nums[:-1])
            for e in temp:
                ans.append(e + [nums[-1]])
            ans += temp
        return ans

    def exist(self, board: List[List[str]], word: str) -> bool:
        """Q79"""
        def exist_helper(i, j, current):
            nonlocal ans
            if ans or visited[i][j] or word[current] != board[i][j]:
                return
            if current == len(word) - 1:
                ans = True
                return
            visited[i][j] = True
            if i > 0:
                exist_helper(i-1, j, current + 1)
            if i < m-1:
                exist_helper(i+1, j, current + 1)
            if j > 0:
                exist_helper(i, j-1, current + 1)
            if j < n-1:
                exist_helper(i, j+1, current + 1)
            visited[i][j] = False

        # if occurrence of left char is greater than right char in board, flip
        board_counter = collections.Counter([c for row in board for c in row])
        for i in range(len(word)//2):
            l, r = board_counter[word[i]], board_counter[word[-i]]
            if l > r:
                word = word[::-1]
            elif l == r:
                continue
            break

        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]
        ans = False

        for i in range(m):
            for j in range(n):
                if ans:
                    return ans
                exist_helper(i, j, 0)
        return ans

    def removeDuplicates(self, nums: List[int]) -> int:
        """Q80"""
        n = len(nums)
        if n <= 2:
            return n
        # [slow - 1] points to last valid pisition, [fast] points to current number to check
        slow = 2
        for fast in range(2, n):
            if nums[fast] != nums[slow - 2]:
                nums[slow] = nums[fast]
                slow += 1
        return slow

    def search(self, nums: List[int], target: int) -> bool:
        """Q81"""
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                return True

            if nums[l] == nums[m] == nums[r]:
                # no: [1, 2, 1]
                # yes: [3, 1, 2, 3, 3, 3, 3]
                l += 1
                r -= 1
            elif nums[l] <= nums[m]:
                # left part ascending
                if nums[l] <= target < nums[m]:
                    # search left
                    r = m - 1
                else:
                    l = m + 1
            else:
                # right part ascending
                if nums[m] < target <= nums[r]:
                    # search right
                    l = m + 1
                else:
                    r = m - 1
        return False

    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """Q82"""
        dummy = ListNode(None, head)
        cursor = dummy
        while cursor.next and cursor.next.next:
            if cursor.next.val != cursor.next.next.val:
                cursor = cursor.next
            else:
                x = cursor.next.val
                # [1, 2, 3, 3] cursor = 2, x = 3, delete all next val = 3
                while cursor.next and cursor.next.val == x:
                    cursor.next = cursor.next.next
        return dummy.next

    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """Q83"""
        dummy = ListNode(None, head)
        cursor = dummy
        while cursor.next:
            cursor = cursor.next
            while cursor.next and cursor.next.val == cursor.val:
                cursor.next = cursor.next.next
        return dummy.next

    def largestRectangleArea(self, heights: List[int]) -> int:
        """Q84"""
        n = len(heights)
        if n == 1:
            return heights[0]

        heights = [0] + heights + [0]
        ans = 0
        stack = [0]  # stores ascending heights
        for i in range(1, n + 2):
            while heights[stack[-1]] > heights[i]:
                # current_height = heights[stack.pop()]
                # current_width = i - 1 - stack[-1]
                ans = max(ans, heights[stack.pop()] * (i - 1 - stack[-1]))
            stack.append(i)
        return ans

    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        """Q85"""
        def maximalRectangle_helper(heights: List[int]) -> int:
            # same as Q84
            temp = heights.copy()
            temp.append(0)
            stack = [-1]
            ans = 0
            for i, height in enumerate(temp):
                while temp[stack[-1]] > height:
                    ans = max(ans, temp[stack.pop()] * (i - 1 - stack[-1]))
                stack.append(i)
            return ans

        ans, n = 0, len(matrix[0])
        current_row = [0] * n
        # calculate rectangle for every row
        for row in matrix:
            current_row = [(current_row[i] + 1) if row[i]
                           == "1" else 0 for i in range(n)]
            ans = max(ans, maximalRectangle_helper(current_row))
        return ans

    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        """Q86"""
        small = ListNode(0, None)
        samll_head = small
        large = ListNode(0, None)
        large_head = large
        while head:
            if head.val < x:
                small.next = head
                small = small.next
            else:
                large.next = head
                large = large.next
            head = head.next
        small.next = large_head.next
        large.next = None
        return samll_head.next

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """Q88"""
        p1 = m-1
        p2 = n-1
        for tail in range(m+n-1, -1, -1):
            if p2 < 0:
                break
            # p1 < 0, go to else
            if p1 >= 0 and nums1[p1] > nums2[p2]:
                nums1[tail] = nums1[p1]
                p1 -= 1
            else:
                nums1[tail] = nums2[p2]
                p2 -= 1

    def grayCode(self, n: int) -> List[int]:
        """Q89"""
        ans = [0]  # n = 1
        for i in range(1, n+1):
            for j in range(len(ans)-1, -1, -1):
                # 00
                # 01
                # 11, add 1
                # 10, add 1
                ans.append(ans[j] | (1 << (i-1)))
        return ans

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """Q90"""
        nums.sort()
        n = len(nums)
        ans = []
        for bin in range(1 << n):
            current = []
            choose = True
            for i in range(n):
                # check if the i-th bit in bin is 1
                if bin & (1 << i) != 0:
                    # bin >> (i-1) & 1 == 0: check if the (i-1)-th bit is 0
                    # this condition means, 如果i和i-1重复, 并且i-1没有被选，则当前跳过
                    # for example, [2, 2, 2, 3]
                    # 000*, 100*, 110*, 111* valid
                    # 101*, 011*, 010*, 001* invalid
                    if i > 0 and nums[i] == nums[i-1] and bin >> (i-1) & 1 == 0:
                        choose = False
                        break
                    current.append(nums[i])
            if choose:
                ans.append(current)
        return ans

    def numDecodings(self, s: str) -> int:
        """Q91"""
        n = len(s)
        dp = [1] + [0] * n
        # i-th char
        for i in range(1, n+1):
            if s[i-1] != "0":
                dp[i] += dp[i-1]
            if i > 1 and s[i-2] != "0" and int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]
        return dp[n]

    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        """Q92"""
        dummy = ListNode(0, head)
        pre = dummy
        for _ in range(left-1):
            pre = pre.next
        current = pre.next
        # pre always points to previous node of the first in interval
        # current points to the first node in interval
        for _ in range(right - left):
            next = current.next
            current.next = next.next
            next.next = pre.next
            pre.next = next
        return dummy.next

    def restoreIpAddresses(self, s: str) -> List[str]:
        """Q93"""
        if len(s) > 12 or len(s) < 4:
            return []
        ans = []
        segments = [0] * 4

        def restoreIpAddresses_helper(segId: int, segStart: int):
            if segId == 4:
                if segStart == len(s):
                    ip = ".".join(map(str, segments))
                    ans.append(ip)
                return

            if segStart == len(s):
                return

            if s[segStart] == "0":
                segments[segId] = 0
                restoreIpAddresses_helper(segId+1, segStart+1)
                return

            addr = 0
            for segEnd in range(segStart, len(s)):
                addr = addr * 10 + int(s[segEnd])
                if 0 < addr <= 255:
                    segments[segId] = addr
                    restoreIpAddresses_helper(segId+1, segEnd+1)
                else:
                    break

        restoreIpAddresses_helper(0, 0)
        return ans

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """Q94"""
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right) if root else []

    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        """Q95"""
        def generateTrees_helper(start, end):
            if start > end:
                return [None]
            ans = []
            for i in range(start, end + 1):
                left = generateTrees_helper(start, i-1)
                right = generateTrees_helper(i+1, end)

                for l in left:
                    for r in right:
                        current = TreeNode(i, l, r)
                        ans.append(current)
            return ans

        return generateTrees_helper(1, n) if n else []

    def numTrees(self, n: int) -> int:
        """Q96"""
        g = [0] * (n+1)
        g[0], g[1] = 1, 1

        for k in range(2, n+1):
            # calculate the k-th
            for i in range(1, k+1):
                g[k] += g[i-1] * g[k-i]
        return g[n]

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        """Q97"""
        m, n, l = len(s1), len(s2), len(s3)
        if m + n != l:
            return False

        # means first i in s1 and fisrt j in s2 can be first (i+j) in s3
        dp = [[False] * (n+1) for _ in range(m+1)]
        dp[0][0] = True
        for i in range(m+1):
            for j in range(n+1):
                p = i+j-1
                if i > 0:
                    # i-th in s1 == (i+j)-th in s3
                    dp[i][j] = dp[i-1][j] and s1[i-1] == s3[p]
                if j > 0 and not dp[i][j]:
                    dp[i][j] = dp[i][j-1] and s2[j-1] == s3[p]
        return dp[m][n]

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """Q98"""
        def isValidBST_helper(node: TreeNode, lower=float("-inf"), upper=float("inf")):
            return True if not node else all([
                node.val > lower,
                node.val < upper,
                isValidBST_helper(node.left, lower, node.val),
                isValidBST_helper(node.right, node.val, upper)
            ])
        return isValidBST_helper(root)

    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """Q99"""
        x, y, pre = None, None, TreeNode(float("-inf"))
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if pre.val > root.val:
                y = root
                if not x:
                    x = pre
                else:
                    break
            pre = root
            root = root.right

        x.val, y.val = y.val, x.val

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """Q100"""
        if not p and not q:
            return True
        elif not p or not q:
            return False
        elif p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """Q101"""
        def isSymmetric_helper(p: TreeNode, q: TreeNode):
            if not p and not q:
                return True
            elif not p or not q:
                return False
            elif p.val != q.val:
                return False
            return isSymmetric_helper(p.left, q.right) and isSymmetric_helper(p.right, q.left)

        return isSymmetric_helper(root.left, root.right)

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """Q102"""
        if not root:
            return []
        ans = []
        current = [root]
        next = []
        while current:
            for e in current:
                if e.left:
                    next.append(e.left)
                if e.right:
                    next.append(e.right)
            ans.append([e.val for e in current])
            current = next
            next = []
        return ans

    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """Q103"""
        if not root:
            return []
        ans = []
        q = [root]
        leftOrder = True
        while q:
            n = len(q)
            temp = []
            for _ in range(n):
                current = q.pop(0)
                temp.append(current.val)
                if current.left:
                    q.append(current.left)
                if current.right:
                    q.append(current.right)
            if not leftOrder:
                temp.reverse()
            ans.append(temp)
            leftOrder = not leftOrder
        return ans

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """Q104"""
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right)) if root else 0

    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """Q105"""
        def buildTree_helper(pre_left, pre_right, in_left, in_right):
            if pre_left > pre_right:
                # empty sub tree
                return None
            pre_root = pre_left
            in_root = index[preorder[pre_root]]

            root = TreeNode(preorder[pre_root])
            size_left = in_root - in_left
            root.left = buildTree_helper(
                pre_left + 1, pre_left + size_left, in_left, in_root - 1)
            root.right = buildTree_helper(
                pre_left + size_left + 1, pre_right, in_root + 1, in_right)
            return root

        index = {e: i for i, e in enumerate(inorder)}
        n = len(preorder)
        return buildTree_helper(0, n-1, 0, n-1)

    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """Q106"""
        def buildTree_helper(in_left, in_right, post_left, post_right):
            if post_left > post_right:
                return None

            post_root = post_right
            in_root = index[postorder[post_root]]
            size_left = in_root - in_left

            root = TreeNode(postorder[post_root])
            root.left = buildTree_helper(
                in_left, in_root-1, post_left, post_left+size_left-1)
            root.right = buildTree_helper(
                in_root+1, in_right, post_left+size_left, post_right-1)
            return root

        index = {e: i for i, e in enumerate(inorder)}
        n = len(inorder)
        return buildTree_helper(0, n-1, 0, n-1)

    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        """Q107"""
        if not root:
            return []
        ans = []
        q = [root]

        while q:
            n = len(q)
            temp = []
            for _ in range(n):
                current = q.pop(0)
                temp.append(current.val)
                if current.left:
                    q.append(current.left)
                if current.right:
                    q.append(current.right)
            ans.insert(0, temp)
        return ans

    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        """Q108"""
        n = len(nums)
        if n == 0:
            return None
        if n == 1:
            return TreeNode(nums[0])
        mid = n // 2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root


# ----------------------------------------------------
s = Solution()
node15 = TreeNode(15)
node7 = TreeNode(7)
node20 = TreeNode(20, node15, node7)
node9 = TreeNode(9)
node3 = TreeNode(3, node9, node20)
print(s.levelOrderBottom(node3))
