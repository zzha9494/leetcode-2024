from typing import List


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
