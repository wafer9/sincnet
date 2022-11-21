1. pre-emphasis (avg_60.pt)
Overall -> 5.02 % N=104765 C=99599 S=5035 D=131 I=92
Mandarin -> 5.02 % N=104762 C=99599 S=5032 D=131 I=92
Other -> 100.00 % N=3 C=0 S=3 D=0 I=0

## 2022.11.21 Result

* Feature info: using sinc feature, sincnet.required_grad = False
* Training info: lr 0.002, batch size 16, 7 gpu, acc_grad 4, 240 epochs

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search avg_30  | 5.10 % N=104765 C=99539 S=5103 D=123 I=118  |
| ctc greedy search avg_60  | 5.09 % N=104765 C=99539 S=5106 D=120 I=111  |