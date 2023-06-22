import torch

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary


import matplotlib.pyplot as plt
import numpy as np

#cuda로 보낸다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#비교대상 선택
print("다음 항목중 비교할 Loss함수의 번호를 두개 입력해주세요")
print("1. MSELoss \n2. SmoothL1Loss \n3. L1Loss  \n4. KLDivLoss \n5. CrossEntropyLoss \n6. myloss")
test1, test2 = map(int, input().split())


print("다음중 원하는 옵티마이저의 번호를 입력해주세요")
print("1. Adam \n2. SGD \n3. Adagrad")
op = input()

print("함수를 반복할 횟수를 입력해주세요")
num = int(input())

#그래프 저장용
loss_graph1 = []
loss_graph2 = []
iter_graph1 = []
iter_graph2 = []


#==========================================
#1) 모델 생성
model = models.vgg16(pretrained=True).to(device)
my_layer2 = nn.Sigmoid()
model.classifier.add_module("7", my_layer2)

print('========= Summary로 보기 =========')
summary(model, (3, 100, 100))
'''
def myloss(result, target):
    return ((torch.sum((result-target)*(result-target)))/10)
'''

#2) loss function
#꼭 아래와 같이 2단계, 클래스 선언 후, 사용
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()

if (test1==1): criterion1 = nn.MSELoss() #(x-y)의 제곱값
elif (test1==2): criterion1 = nn.SmoothL1Loss() #
elif (test1==3): criterion1 = nn.L1Loss() #(x-y)의 절대값
elif (test1==4): criterion1 = nn.KLDivLoss() #y * (log y -x)
elif (test1==5): criterion1 = nn.CrossEntropyLoss() #-sigma x logy


if (test2==1): criterion2 = nn.MSELoss()
elif (test2==2): criterion2 = nn.SmoothL1Loss()
elif (test2==3): criterion2 = nn.L1Loss()
elif (test2==4): criterion2 = nn.KLDivLoss()
elif (test2==5): criterion2 = nn.CrossEntropyLoss()


#3) activation function
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if (op ==1): optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
elif (op==2): optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
elif (op==2): optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)


#모델이 학습 모드라고 알려줌
model.train()

for j in range(num):
    #옵티마이저 초기화
    optimizer.zero_grad()

    #입력값 생성하고
    a = torch.randn(12,3,100,100).to(device)

    #모델에 넣은다음
    result = model(a)

    #결과와 동일한 shape을 가진 Ground-Truth 를 읽어서
    #target  = torch.randn_like(result)

    #타겟값을 1로 바꾸어서 네트워크가 무조건 1만 출력하도록 만든다.
    target = torch.ones_like(result)

    #네트워크값과의 차이를 비교
    if(test1 ==6): loss1=((torch.sum((result-target)*(result-target)))/10).to(device)
    else: loss1 = criterion1(result, target).to(device)

    loss_graph1.append(loss1.item())
    iter_graph1.append(j)

    #=============================
    #loss는 텐서이므로 item()
    print("epoch: {} loss:{} ".format(j, loss1.item()))

    #loss diff값을 뒤로 보내서 grad에 저장하고
    loss1.backward()

    #저장된 grad값을 기준으로 activation func을 적용한다.
    optimizer.step()

for k in range(num):
    # 옵티마이저 초기화
    optimizer.zero_grad()

    # 입력값 생성하고
    a = torch.randn(12, 3, 100, 100).to(device)

    # 모델에 넣은다음
    result1 = model(a)

    # 결과와 동일한 shape을 가진 Ground-Truth 를 읽어서
    # target  = torch.randn_like(result)

    # 타겟값을 1로 바꾸어서 네트워크가 무조건 1만 출력하도록 만든다.
    target1 = torch.ones_like(result1)

    # 네트워크값과의 차이를 비교
    if(test2 == 6):  loss2 =((torch.sum((result1-target1)*(result1-target1)))/10).to(device)
    else:
        loss2 = criterion2(result1, target1).to(device)

    loss_graph2.append(loss2.item())
    iter_graph2.append(k)

    # =============================
    # loss는 텐서이므로 item()
    print("epoch: {} loss:{} ".format(k, loss2.item()))

    # loss diff값을 뒤로 보내서 grad에 저장하고
    loss2.backward()

    # 저장된 grad값을 기준으로 activation func을 적용한다.
    optimizer.step()

fig, axs = plt.subplots(1,2)
axs[0].plot(iter_graph1, loss_graph1, '-b')
axs[0].set_title('loss1 graph')
axs[0].set_ylabel('loss')
axs[1].plot(iter_graph2, loss_graph2, '--b')
axs[1].set_title('loss2 graph')
axs[1].set_ylabel('loss')
plt.savefig("result6.png")  # should before show method
plt.show()


'''
import torch

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary


import matplotlib.pyplot as plt
import numpy as np

op = torch.tensor([[0.8982, 0.805, 0.6393, 0.9983, 0.5731, 0.0469, 0.556, 0.1476, 0.8404,0.5544]])
tg = torch.LongTensor([0,1,0,0,0,0,0,0,0,0])
cri = nn.MSELoss()
loss = cri(op, tg)
print(loss)
loss1 = 0
loss1 = (torch.sum((tg-op)*(tg-op)))/10
print(loss1)
'''