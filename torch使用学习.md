#### torch.stack()
- 在维度上连接若干个张量（这些张量形状相同）
- 将若干个张量在dim维度上连接,生成一个扩维的张量，比如说原来你有若干个2维张量，连接可以得到一个3维的张量。
- 设待连接张量维度为n，dim取值范围为-n-1~n，这里得提一下为负的意义：-i为倒数第i个维度。举个例子，对于2维的待连接张量，-1维即3维，-2维即2维
```python
a = torch.randn([3, 3])
b = torch.randn([3, 3]) + 2
c = torch.randn([3, 3]) * 2
print(a)
print(b)
print(c)
print(torch.stack([a, b, c], dim=0))
print(torch.stack([a, b, c], dim=1))
print(torch.stack([a, b, c], dim=2))
print(torch.stack([a, b, c], dim=-1) == torch.stack([a, b, c], dim=2))
```
#### torch.clamp()
- torch.clamp(input, min, max, out=None) → Tensor  将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
- input(Tensor)：输入张量
- min：限制范围下限
- max:限制范围上限
- out(Tensor):输出张量
```python
a = torch.randint(low=0, high=10, size=[10, 2])
    print(a)
    print(torch.clamp(a, min=3, max=8))
```

#### F.interpolate() 上下采样函数
- maxpooling降采样操作是不可逆的
