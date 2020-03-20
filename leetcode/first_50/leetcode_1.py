class num_1:
    """两数之和"""
    def twoSum(self, nums, target) -> list:
        for index, temp in enumerate(nums):
            gap = target - temp
            if gap in nums[index+1:]:
                return sorted([index, nums[index+1:].index(gap)+index+1])
        return []

    def better(self, nums, target) -> list:
        mapping = {}
        for index, temp in enumerate(nums):
            if temp in mapping:
                return sorted([mapping[temp], index])
            gap = target - temp
            mapping[gap] = index
        return None

    @staticmethod
    def check():
        obj = num_1()
        # nums = [2,7,11,15]
        # target = 9
        nums = [3, 2, 4]
        target = 6
        result = obj.twoSum(nums, target)
        print(result)
        better_result = obj.better(nums, target)
        print(better_result)




if __name__ == '__main__':
    num_1.check()


