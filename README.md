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

## 2023.02.11

* Feature info: using sinc feature, sincnet.required_grad = True
* Training info: lr 0.002, batch size 16, 8 gpu, acc_grad 4, 240 epochs

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search avg_30  | 5.15 % N=104765 C=99468 S=5154 D=143 I=98  |
| ctc greedy search avg_60  | 5.14 % N=104765 C=99476 S=5151 D=138 I=95  |

## 2023.02.13

* Feature info: using sinc feature, sincnet.required_grad = True
* Training info: lr 0.002, batch size 16, 8 gpu, acc_grad 4, 240 epochs

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search avg_30  | 5.10 % N=104765 C=99525 S=5084 D=156 I=106 |
| ctc greedy search avg_60  | 5.05 % N=104765 C=99575 S=5035 D=155 I=99  |


## 2023.02.16
* Feature info: using sinc feature, sincnet.required_grad = True
* Training info: lr 0.002, batch size 16, 8 gpu, acc_grad 4, 240 epochs, num_f_mask = 2

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search   avg_30  | 4.98 % N=104765 C=99647 S=4982 D=136 I=101 |
| ctc greedy search   avg_60  | 4.98 % N=104765 C=99653 S=4971 D=141 I=101 |
| attention           avg_30  | 5.03 % N=104765 C=99622 S=4979 D=164 I=127 |
| attention_rescoring avg_30  | 4.71 % N=104765 C=99916 S=4722 D=127 I=87  |


## 2023.02.17
* Feature info: using sinc feature, sincnet.required_grad = True
* Training info: lr 0.002, batch size 16, 8 gpu, acc_grad 4, 240 epochs, num_f_mask = 2
  add global_cmvn

| decoding mode               |         CER                                |
|-----------------------------|--------------------------------------------|
| ctc greedy search   avg_30  | 4.99 % N=104765 C=99636 S=4982 D=147 I=103 |
| ctc greedy search   avg_60  | 4.95 % N=104765 C=99677 S=4948 D=140 I=103 |
| attention           avg_30  | 4.91 % N=104765 C=99725 S=4844 D=196 I=102 |
| attention_rescoring avg_30  | 4.61 % N=104765 C=100025 S=4615 D=125 I=91 |


## 2023.02.18
* Feature info: using sinc feature, sincnet.required_grad = True
* Training info: lr 0.002, batch size 16, 8 gpu, acc_grad 4, 240 epochs, num_f_mask = 2
  add global_cmvn,  rm Concat Liner,

| decoding mode               |         CER                                |
|-----------------------------|--------------------------------------------|
| ctc greedy search   avg_30  | 4.98 % N=104765 C=99650 S=4975 D=140 I=104 |
| ctc greedy search   avg_60  | 4.97 % N=104765 C=99665 S=4956 D=144 I=103 |
| attention           avg_30  | 4.98 % N=104765 C=99659 S=4926 D=180 I=112 |
| attention_rescoring avg_30  | 4.70 % N=104765 C=99935 S=4702 D=128 I=92  |


## 2023.02.23
* Feature info: using sinc feature, sincnet.required_grad = True
* Training info: lr 0.002, batch size 16, 8 gpu, acc_grad 4, 240 epochs, num_f_mask = 2
  add global_cmvn,  rm Concat Liner, use stft instead of bandstop-filter

| decoding mode               |         CER                                |
|-----------------------------|--------------------------------------------|
| ctc greedy search   avg_30  | 4.93 % N=104765 C=99692 S=4931 D=142 I=97  |
| ctc greedy search   avg_60  | 4.94 % N=104765 C=99691 S=4932 D=142 I=97  |
| attention           avg_30  | 4.91 % N=104765 C=99721 S=4849 D=195 I=104 |
| attention_rescoring avg_30  | 4.60 % N=104765 C=100036 S=4597 D=132 I=85 |