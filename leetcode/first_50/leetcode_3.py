class num_3:
    # https: // zhuanlan.zhihu.com / p / 105476748
    # 无重复字符的最长子串， 利用滑动窗口，时间复杂度为O(n)
    def lengthOfLongestSubstring(self, s: str) -> int:
        lookup = set()
        max_len = 0
        cur_len = 0
        left = 0

        for index in range(len(s)):
            while (s[index] in lookup):
                lookup.remove(s[left])
                cur_len -= 1
                left += 1
            lookup.add(s[index])
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
        return max_len

    def templet(self, s) -> list:
        from collections import defaultdict
        lookup = defaultdict(int)
        start = 0
        end = 0
        max_len = 0
        counter = 0
        while end < len(s):
            if lookup[s[end]] > 0:
                counter += 1
            lookup[s[end]] += 1
            end += 1
            while counter > 0:
                if lookup[s[start]] > 1:
                    counter -= 1
                lookup[s[start]] -= 1
                start += 1
            max_len = max(max_len, end - start)
        return max_len

    @staticmethod
    def check():
        obj = num_3()
        a = 'abcabcbb'
        result = obj.lengthOfLongestSubstring(a)
        print(result)
        better_result = obj.templet(a)
        print(better_result)




if __name__ == '__main__':
    num_3.check()



