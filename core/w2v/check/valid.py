#!/usr/bin/python
# -*- coding: utf-8 -*-


import torch



input = torch.randn(2, 3, 4)
print(input)
print()
mat2 = torch.randn(2, 4, 5)
print(mat2)
print()
res = torch.bmm(input, mat2)
print(res)
print(res.size())