import collections
from functools import lru_cache
import itertools
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

        def helper(current_digits, current_combination):
            if len(current_digits) == 0:
                ans.append(current_combination)
            else:
                for letter in dic[current_digits[0]]:
                    helper(current_digits[1:], current_combination + letter)

        ans = []
        helper(digits, "")
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
        return self.mergeKLists_helper(merged_left, merged_right)

    def mergeKLists_helper(self, left: ListNode, right: ListNode):
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
