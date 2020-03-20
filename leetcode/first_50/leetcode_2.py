import json
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class num_2:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        input_1 = []
        input_2 = []
        next_node = l1
        while (next_node):
            input_1.append(str(next_node.val))
            next_node = next_node.next
        next_node = l2
        while (next_node):
            input_2.append(str(next_node.val))
            next_node = next_node.next
        input_1.reverse()
        input_2.reverse()
        sum = list(str(int(''.join(input_1)) + int(''.join(input_2))))
        start = ListNode(sum[-1])
        now = start
        sum.reverse()
        for temp in sum[1:]:
            _v = ListNode(temp)
            now.next = _v
            now = _v
        return start

    def better(self, l1: ListNode, l2: ListNode) -> ListNode:
        first = ListNode(0)
        carry = 0
        cur_node = first
        while(l1 is not None or l2 is not None):
            input_1 = 0 if l1 is None else l1.val
            input_2 = 0 if l2 is None else l2.val

            sum = input_1 + input_2 + carry
            carry = sum // 10
            new_node = ListNode(sum % 10)
            cur_node.next = new_node
            cur_node = cur_node.next
            if l1 is not None:
                l1 = l1.next
            if l2 is not None:
                l2 = l2.next
        if carry > 0:
            cur_node.next = ListNode(carry)
        return first.next


    @classmethod
    def check(cls):
        node_1 = ListNode(2)
        node_2 = ListNode(4)
        node_3 = ListNode(3)
        node_1.next = node_2
        node_2.next = node_3

        node_4 = ListNode(5)
        node_5 = ListNode(6)
        node_6 = ListNode(7)
        node_4.next = node_5
        node_5.next = node_6

        l1 = node_1
        l2 = node_4
        obj = num_2()
        result = obj.addTwoNumbers(l1, l2)
        print(result)
        better_result = obj.better(l1, l2)
        print(better_result)

if __name__ == '__main__':
    num_2.check()