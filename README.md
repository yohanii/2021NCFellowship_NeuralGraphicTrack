

# 2021 NC Fellowship Neural Graphic Track



## 목표

노이즈가 포함된 마커 데이터로 부드럽고 깨끗한 스켈레톤 애니메이션을 추정하는 것.

입력 데이터는 C3D에서 추출한 마커들의 Position 데이터

출력 데이터는 BVH에서 추출한 관절들의 Euler angle 데이터에서 변환된 Rotation Matrix 데이터



#### 학습데이터

CMU Graphics Lab Motion Capture Database

Carnage Mellon Graphics Lab에서 모션 캡쳐 연구를 위해 촬영된 데이터셋

다양한 포맷으로 제공되어 관련 연구에서 가장 많이 사용되는 모션 데이터셋

크게 6개의 카테고리, 23개의 서브 카테고리, 총 2605개의 대규모 데이터셋으로 제공



## 진행상태



**기존 코드**

```
00001 | Train Loss: 0.3653895 | Valid Loss: 0.5380832 | lr: 0.0009900 | time: 3.608

00101 | Train Loss: 0.0452509 | Valid Loss: 0.4892339 | lr: 0.0003624 | time: 3.586
```

 

**5_CNN(MSE loss)**

```
########## Start Train ##########

 00001 | Train Loss: 0.4119846 | Valid Loss: 0.5723141 | lr: 0.0009900 | time: 100.328

Model saved.

Result saved. result/subject_5_CNN/000001_result_valid

00100 | Train Loss: 0.2345810 | Valid Loss: 0.5939790 | lr: 0.0003660 | time: 100.324

Model saved.

Result saved. result/subject_5_CNN/000100_result_valid

 00101 | Train Loss: 0.2345296 | Valid Loss: 0.5887196 | lr: 0.0003624 | time: 100.166
```

 

 

**5_CNN(CrossEntropyloss)**

```
00001 | Train Loss: 0.4210672 | Valid Loss: 0.5921927 | lr: 0.0009900 | time: 11.724

00100 | Train Loss: 0.2353331 | Valid Loss: 0.5723641 | lr: 0.0003660 | time: 11.859
```



**5_customCNN (loss fun : CrossEntropyloss) 망한 버전.**

```
00001 | Train Loss: -79032341286695568.0000000 | Valid Loss: 4050498143409622.5000000 | lr: 0.0009900 | time: 19.653

Result saved. result/subject_5_CNN/000100_result_valid

00101 | Train Loss: -11946580551261462642819072.0000000 | Valid Loss: 116507478109352062091264.0000000 | lr: 0.0003624 | time: 19.712
```

 

**5_customCNN (loss fun : CrossEntropyloss)**

```
00001 | Train Loss: 0.4041500 | Valid Loss: 0.5810984 | lr: 0.0009900 | time: 19.356

00101 | Train Loss: 0.1842482 | Valid Loss: 0.5606981 | lr: 0.0003624 | time: 19.504
```

Linear 보다는 성능 좋고, 기존 CNN과는 비슷

이 중에선 RNN이 가장 부드럽고 좋은 성능을 냄.

 

**5_ResNet(MSE loss)**

```
00001 | Train Loss: 0.5491642 | Valid Loss: 0.5714595 | lr: 0.0009900 | time: 126.209

00101 | Train Loss: 0.2076657 | Valid Loss: 0.5751043 | lr: 0.0003624 | time: 122.598
```

CNN모델 중에서는 가장 나은 모습.

그러나 노이즈가 엄청 심함.

 

**5_VGG(Cross Entropy loss)**

```
00001 | Train Loss: -38528265133739627968987136.0000000 | Valid Loss: 978813578848653325369344.0000000 | lr: 0.0009900 | time: 54.695

Result saved. result/subject_5_VGG/000100_result_valid

 00101 | Train Loss: -5606094521341963530619846656.0000000 | Valid Loss: 72911864223186672200712192.0000000 | lr: 0.0003624 | time: 54.876
```

망함

 

**5_VGG(MSE loss)**

```
00001 | Train Loss: 352.1761556 | Valid Loss: 1.6048460 | lr: 0.0009900 | time: 54.451

Result saved. result/subject_5_VGG/000100_result_valid

00101 | Train Loss: 1.0000445 | Valid Loss: 0.7763272 | lr: 0.0003624 | time: 54.904
```

망함. 결과 이상

 

#### **12/29 현재 상황**

제공해준 3가지방법(linear, cnn, rnn)과 customCNN, Resnet, VGG에 대해 돌려봄.

이 중 rnn과 resnet의 결과가 가장 좋았다. CrossEntropyloss써보니 망하더라 -> classification용도라 사용하면 안됐음. -> loss fun 생각 좀 하고 고르자

VGG는 아예 망함 -> 무작정 대입하지 말고, dataset에 대해 이해하고 적용시키자

 

앞으로

Robust논문 분석해서 그 모델과 loss fun을 결정하자.

Marker의 index가 어떻게 되는지 파악하고 index마다 accuracy를 분석해 정확도 낮은 index 제거하는 방향.

Data Augmentation을 통해 기존 데이터를 살짝 회전 또는 살짝 이동시켜서 train data양을 늘린다.

Output을 31개의 관절 position matrix와 root position을 따로 학습시키는건 어떤가?

의미가 있을까?

 

무작위로 노드를 제거하는 Dropout을 쓰면 노이즈가 줄지 않을까? -> 사용해보기

 

**5_customCNN (loss fun : MSE)**

```
Result saved. result/subject_5_customCNN/000100_result_valid

00101 | Train Loss: 0.1813815 | Valid Loss: 0.5848318 | lr: 0.0003624 | time: 19.500

Result saved. result/subject_5_customCNN/000100_result_valid

00101 | Train Loss: 0.1813815 | Valid Loss: 0.5848318 | lr: 0.0003624 | time: 19.500
```

노이즈 너무 심함.

 

#### **1/12일 현재 상황**

Robust 논문에서 corrupt이후가 내 상황인 거 같음.

따라서 신경망 모델만 가져옴

Run4는 dense layer으로 이루어진 5개의 residual block으로 구성된 모델 사용.

He 초깃값 사용, 논문에서는 Lecun Initialization 사용했음.

Loss는 아직 MSE -> 바꾸기

Adam -> AdamW의 amsgrad 알고리즘 사용 

 

Run4는 valid loss 전혀 줄어들지 않는 모습.

![img](file:///C:/Users/Yohan/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

![img](file:///C:/Users/Yohan/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

 

![img](file:///C:/Users/Yohan/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg)



#### 1/26 현재 상황

3차 모임 전까지 (x,y,z,c)에 대해 Normalization 시도해 봤음.

결과는 큰 영향을 미치지 않았고, valid loss 또한 줄지 않는 모습을 보임.



3차 모임을 통해 다른 팀들의 진행 상황과 아이디어, 앞으로 해야할 일들을 배울 수 있었음.



이후 41개의 마커 위치 파악함.

<img src="C:\Users\Yohan\Documents\카카오톡 받은 파일\KakaoTalk_20220126_210805398.jpg" style="zoom:50%;" />



##### 해야할 일들

- 전처리
  1. 로컬 좌표 상에서 마커 표현
  2. 마커 스케일링

- Data Argumentation
  1. Occlusion 상황
  2. Noise 있는 상황

- Output을 9x9 matrix가 아닌 4개로 줄일 것
- 후처리로 Noise 제거

(+) 부위별로 나눠서 입력으로 넣을 때 Encoder로 처리하는 이유 알아보기





#### 01/29 현재 상황

**Run4_3(Robust)**

```
00001 | Train Loss: 229479.8873847 | Valid Loss: 0.7170390 | lr: 0.0009900 | time: 40.909

00101 | Train Loss: 1.2996938 | Valid Loss: 0.4163985 | lr: 0.0003624 | time: 41.242
```

17개 노드만 뽑아서 로컬 좌표로 한 거.

다만, 월드좌표로 변환 안함.

Valid loss 줄긴 했지만, 애니메이션은 제대로 안나옴.

Normalization 해야할 듯, 월드좌표로 변환할 것.

 

#### 01/30 현재 상황

Root 좌표는 글로벌

17개 노드는 로컬

스케일링 : 마커데이터를 허벅지 길이를 기준으로 다 비슷하게 맞춰보자

è 허벅지 길이를 1로 설정 : 인덱스 10, 11

노말리제이션 : 형태 무너지지 않게 진행

아웃풋 데이터 줄이기

Loss function 생각

중간과정에도 Loss 넣을 수 있다.

 

 

 

 

 

 

 

**Run4_4**

```
00001 | Train Loss: 1.0875892 | Valid Loss: 0.2598303 | lr: 0.0009900 | time: 40.934

00101 | Train Loss: 0.0044409 | Valid Loss: 0.2103231 | lr: 0.0003624 | time: 41.697
```

스케일링 한 버전

Valid_loss 0.22정도로 떨어진 효과

애니메이션은 행동은 적당히 잘 따라하지만, Position이 불안정함

Root 포지션에 Loss를 쎄게 걸어주자.

노말레이션 일단 스킵하자.

Root 포지션으로 결과값 0,1,2 예측하고

상대좌표로 나머지 예측

Loss 도 따로 걸기

리지드 단위로 쪼개서 일정 거리이상 멀어지면 노이즈로 판단해서 제거해주기 추가.

 

#### 02/01 현재 상황

**Run4_6**

```
00001 | Train Loss: 276427417.6532329 | Valid Loss: 14.5412589 | lr: 0.0009900 | time: 80.792

00001 | Train Loss: 289867.5554182 | Valid Loss: 4.7227690 | lr: 0.0009900 | time: 80.792

00028 | Train Loss: 150.7324746 | Valid Loss: 7.4777530 | lr: 0.0007547 | time: 80.866

00028 | Train Loss: 2.3765926 | Valid Loss: 2.5457294 | lr: 0.0007547 | time: 80.866
```

Root 포지션과 상대좌표를 분리해서 학습

X,y,z 축을 y,z,x 축으로 바꿔서 진행하였음

Result2의 Valid loss가 매우 크게 나타나 이건 기존대로 해봐야할 듯

알고보니 스케일링 안된 상태 였음.

 

 

 

**Run4_7**

```
00001 | Train Loss: 234080160.8488746 | Valid Loss: 15.7547968 | lr: 0.0009900 | time: 81.001

00001 | Train Loss: 0.2596929 | Valid Loss: 0.1110346 | lr: 0.0009900 | time: 81.001

00101 | Train Loss: 42.9625089 | Valid Loss: 2.2911412 | lr: 0.0003624 | time: 81.680

00101 | Train Loss: 0.0024128 | Valid Loss: 0.0786881 | lr: 0.0003624 | time: 81.680
```

**Score| Distance: 3.5454, Smoothness: 0.0638**

Result1은 축 바꾼 버전

Result2는 축 안 바꾸고 스케일링은 된 버전.

Root pos만 스케일링 잘 하면 많이 안정화 될 듯 보임.

![img](file:///C:/Users/Yohan/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)

일단 train_input의 Root pos는 0~74165까지 왼쪽의 작은 범위, 74166~104544까지 오른쪽은 큰 범위를 가지고 왼쪽 범위에 섞여 있기도 함.

이걸 직접 분리해서 스케일링을 해줘서 넣어줄 수는 있으나 자동화가 되는 것이 아니라 비슷한 문제가 발생했을 때 알아서 대처할 수 없다는 점이 문제이다.

그래도 일단 그렇게 진행하고 다른 부분을 계속 개선한 후 최종 데이터가 나오면 그 때 다시 대처하는 식으로 진행하자.

 

Y[index]값이 1000을 넘냐 안 넘냐로 나눠서 따로 스케일링 해주자.

 

**Run4_8**

```
00001 | Train Loss: 292161547.6444343 | Valid Loss: 14.6895159 | lr: 0.0009900 | time: 81.699

00001 | Train Loss: 0.4070729 | Valid Loss: 0.5575088 | lr: 0.0009900 | time: 81.699

Result saved. result/subject_5_Run4_8/000001_result_valid

00002 | Train Loss: 3257.4694122 | Valid Loss: 14.8933397 | lr: 0.0009801 | time: 81.784

00002 | Train Loss: 0.0319619 | Valid Loss: 0.5706395 | lr: 0.0009801 | time: 81.784
```

Result1, Result2 축 바꾼 버전,

Result2 스케일링 된 버전

역시 Result2는 축 안 바꾸는게 맞다.

Result1은 축 바꾸는게 맞는지 아직 모름.

 

```
00001 | Train Loss: 273607885.0396925 | Valid Loss: 15.0856275 | lr: 0.0009900 | time: 81.527

00001 | Train Loss: 0.2743460 | Valid Loss: 0.1176005 | lr: 0.0009900 | time: 81.527
```

Result1, Result2 둘다 축 안바꿈, 

Result2만 스케일링 된 상태

 

**Run4_8**

```
00001 | Train Loss: 30962.3193433 | Valid Loss: 15.4490389 | lr: 0.0009900 | time: 81.378

00001 | Train Loss: 0.2443191 | Valid Loss: 0.1266598 | lr: 0.0009900 | time: 81.378

00035 | Train Loss: 4.7245398 | Valid Loss: 8.6373346 | lr: 0.0007034 | time: 81.290

00035 | Train Loss: 0.0033642 | Valid Loss: 0.0780772 | lr: 0.0007034 | time: 81.290
```

Result1 축 바꿈, 스케일링 됨.

Result2 축 안바꿈, 스케일링 됨.

Result1에서 Train loss는 4_7에 비해 작으나 valid_loss는 8근처에서 정체됨.

오히려 안좋은 효과

Result2는 이전과 동일

 

è Result1 축 바꾸지 말고 돌려보자

 

#### 02/02 현재 상황

**Run4_9**

```
00001 | Train Loss: 32826.5241817 | Valid Loss: 100.0970620 | lr: 0.0009900 | time: 79.710

00001 | Train Loss: 0.2440910 | Valid Loss: 0.1194742 | lr: 0.0009900 | time: 79.710

00010 | Train Loss: 9.0406240 | Valid Loss: 55.2338788 | lr: 0.0009044 | time: 79.642

00010 | Train Loss: 0.0060691 | Valid Loss: 0.0863437 | lr: 0.0009044 | time: 79.642
```

Result1 축 안 바꿈, 스케일링 됨.

Result2 축 안 바꿈, 스케일링 됨.

Result1의 train_loss는 축을 바꿨을 때와 비슷하거나 좀 더 빠르게 줄어드는 거 같으나, valid_loss는 55정도에서 내려가지 못함. 다른 경우에 case 1부터 15인 것에 비하면 아주 나쁜 결과를 보여줌.

 

문제점이 무엇인지 잘 파악하기 힘들다.

스케일링을 했음에도 효과가 좋지 않음.

è 알고보니, train_label의 범위에 다 맞췄음. valid_label의 범위를 고려안함.

Valid_rootnp는 valid_label의 범위에 다시 맞춰주자

 

**Run4_10**

```
00001 | Train Loss: 48939.4996122 | Valid Loss: 11.3468246 | lr: 0.0009900 | time: 79.772

00001 | Train Loss: 0.2918056 | Valid Loss: 0.1219149 | lr: 0.0009900 | time: 79.772

00030 | Train Loss: 5.5192141 | Valid Loss: 5.3492854 | lr: 0.0007397 | time: 80.046

00030 | Train Loss: 0.0043219 | Valid Loss: 0.0774558 | lr: 0.0007397 | time: 80.046
```

Result1 축 안 바꿈, 스케일링 됨

Result2 축 안 바꿈, 스케일링 됨.

여전히 Valid_loss가 5밑으로는 줄지 못함.

 

**Run4_11**

```
00001 | Train Loss: 28315.8962738 | Valid Loss: 3.4301427 | lr: 0.0009900 | time: 80.038

00001 | Train Loss: 0.2993424 | Valid Loss: 0.1162291 | lr: 0.0009900 | time: 80.038

00021 | Train Loss: 6.2655660 | Valid Loss: 6.5659532 | lr: 0.0008097 | time: 80.017

00021 | Train Loss: 0.0050651 | Valid Loss: 0.0809458 | lr: 0.0008097 | time: 80.017
```

Result1 축 바꿈, 스케일링 됨

Result2 축 안 바꿈, 스케일링 됨.

첫번째 시도일 때 valid loss 낮게 나왔으나,

다음 시도부터 다시 5이상에서 머뭄.

현재까지 4_7이 가장 좋은 결과를 보임

즉, Root pos는 축 바꾸고, 스케일링 안될 때

상대좌표는 축 안 바꾸고, 스케일링 되었을 때.

 

**Run4_12**

```
00001 | Train Loss: 102.7104213 | Valid Loss: 5.3904046 | lr: 0.0009900 | time: 80.096

00001 | Train Loss: 0.3213947 | Valid Loss: 0.1162815 | lr: 0.0009900 | time: 80.096

00020 | Train Loss: 1.0213051 | Valid Loss: 5.4207871 | lr: 0.0008179 | time: 80.283

00020 | Train Loss: 0.0051360 | Valid Loss: 0.0811315 | lr: 0.0008179 | time: 80.283
```

**Score| Distance: 8.8166, Smoothness: 0.9854**

Normalization 적용

Normalization은 rootnp는 (104545, 32)개의 x,y,z 각각 따로 담아서 mean, std구하고 직접 적용함.

나머진 4_11과 동일

Train_loss는 많이 줄었는데, valid_loss는 5밑으로 가질 못함.

 

내 생각엔 지금 Result1은 Root position만으로 학습을 하고 있기 때문에 데이터가 너무 적어서 학습이 잘 안되는 거 같음. Root pos주변의 다른 노드들도 추가 제공하면 더 잘 될 거 같음.





#### 02/05 현재 상황

- 노드 17개로 Result1 학습하기

**Run5_1**

```
00001 | Train Loss: 109.6253226 | Valid Loss: 7.8765621 | lr: 0.0009900 | time: 81.164 00001 | Train Loss: 0.2633119 | Valid Loss: 0.1236772 | lr: 0.0009900 | time: 81.164

00018 | Train Loss: 0.0413682 | Valid Loss: 8.4991742 | lr: 0.0008345 | time: 81.612 00018 | Train Loss: 0.0054493 | Valid Loss: 0.0813112 | lr: 0.0008345 | time: 81.612
```

**Score| Distance: 15.3672, Smoothness: 0.2804**

valid_loss가 8밑으로 내려가지 못하는 결과를 보임. 별로 좋지 않음.



- 몸체 근처 노드들로 Result1 학습하기

**Run5_2**

```
00001 | Train Loss: 298952071.9438537 | Valid Loss: 15.7372591 | lr: 0.0009900 | time: 80.980 

00001 | Train Loss: 0.9509046 | Valid Loss: 0.1312283 | lr: 0.0009900 | time: 80.980

00101 | Train Loss: 133.0084980 | Valid Loss: 2.7543883 | lr: 0.0003624 | time: 81.921

00101 | Train Loss: 0.0041868 | Valid Loss: 0.0804653 | lr: 0.0003624 | time: 81.921
```

**Score| Distance: 4.2000, Smoothness: 0.0808**

body_train_input, body_valid_input 사용 : 몸체 부분의 9, 28, 39, 4, 23, 40 노드까지 총 6개의 노드를 의미한다.

스케일링 안한 상태. 

아직까진 Run4_7의 결과가 가장 좋음.

스케일링 해서 학습시켜보자. 



**Run5_3**

```
00001 | Train Loss: 138953710749676.8281250 | Valid Loss: 9.0268844 | lr: 0.0099000 | time: 81.705 

00001 | Train Loss: 361825.2542290 | Valid Loss: 0.1531254 | lr: 0.0099000 | time: 81.705

00027 | Train Loss: 214.9490688 | Valid Loss: 8.3227856 | lr: 0.0076234 | time: 81.857 

00027 | Train Loss: 0.0539082 | Valid Loss: 0.1494122 | lr: 0.0076234 | time: 81.857
```

Init_lr을 0.001에서 0.1로 바꿔봤음.

별로 효과적이진 않고 기존 방식이 더 좋다.



**Run5_4**

```
00001 | Train Loss: 32240.2807486 | Valid Loss: 5.1508511 | lr: 0.0009900 | time: 81.675 

00001 | Train Loss: 0.3321867 | Valid Loss: 0.1189512 | lr: 0.0009900 | time: 81.675

00101 | Train Loss: 0.1348799 | Valid Loss: 6.1420379 | lr: 0.0003624 | time: 82.079 

00101 | Train Loss: 0.0028059 | Valid Loss: 0.0772134 | lr: 0.0003624 | time: 82.079
```

**Score| Distance: 8.4615, Smoothness: 0.0574**

Result1에 스케일링 적용함. 

valid_loss는 여전히 5밑으로는 안 떨어짐.

어떻게 position을 안정화 시켜야할지 잘 모르겠다.

Normalization하고 학습시켜 본 뒤 일단 noise를 제거하는 방향으로 진행한 뒤 다시 돌아와야겠다. 



**Run5_5**

```
00001 | Train Loss: 101.4865650 | Valid Loss: 6.8892547 | lr: 0.0009900 | time: 79.554 

00001 | Train Loss: 0.3401721 | Valid Loss: 0.1266835 | lr: 0.0009900 | time: 79.554

00014 | Train Loss: 0.0192677 | Valid Loss: 6.9768563 | lr: 0.0008687 | time: 80.515 

00014 | Train Loss: 0.0076986 | Valid Loss: 0.0883993 | lr: 0.0008687 | time: 80.515
```

Result1에 Normalization 적용

별로 효과 없음.



**Run5_6**

```
00001 | Train Loss: 258604164.9966089 | Valid Loss: 16.6153912 | lr: 0.0009900 | time: 80.420 

00001 | Train Loss: 0.2610411 | Valid Loss: 0.1295726 | lr: 0.0009900 | time: 80.420

00101 | Train Loss: 142.3462216 | Valid Loss: 2.8292074 | lr: 0.0003624 | time: 80.655 

00101 | Train Loss: 0.0025975 | Valid Loss: 0.0796250 | lr: 0.0003624 | time: 80.655
```

**Score| Distance: 4.2925, Smoothness: 0.0861**

Result1 축 바꿈(x,y,z -> z,x,y) 여태까진 x, y, z -> y, z, x로 바꿨는데 z,x,y로 바꾸는게 맞는 거 같음.

스케일링 적용 안함. 스케일링 안했을 때 결과가 제일 잘나오고 있음.



**Run5_7**

```
00001 | Train Loss: 27479.1731430 | Valid Loss: 12.4168775 | lr: 0.0009900 | time: 79.938 

00001 | Train Loss: 0.2718400 | Valid Loss: 0.1234517 | lr: 0.0009900 | time: 79.938

00035 | Train Loss: 0.3590275 | Valid Loss: 15.6607979 | lr: 0.0007034 | time: 80.077

00035 | Train Loss: 0.0038107 | Valid Loss: 0.0795077 | lr: 0.0007034 | time: 80.077
```

Result1 축 바꿈

스케일링 적용

축 안바꿨을 때 보다 성능 안 좋은 것이 확인됨.

-> train과 valid 둘 다 축을 바꾸는게 맞는 건지 봐야할 듯.

-> 확인 해 봤는데 축을 제대로 바꾼 거 같은데 효과가 없는지 이유를 모르겠음.

-> 다시 확인 결과 기존 축 바꾸는 방식이 맞다. x, y, z -> y, z, x



**Run5_8**

rootnp, valid_rootnp를 x,y,z -> z, x, y로 축 바꿔서 해봄. -> 아닌듯.



**Run5_9**

```
00001 | Train Loss: 348164293.0669429 | Valid Loss: 14.9272291 | lr: 0.0009900 | time: 79.832 

00001 | Train Loss: 0.3130751 | Valid Loss: 0.1254250 | lr: 0.0009900 | time: 79.832

00101 | Train Loss: 134.2194982 | Valid Loss: 1.5915595 | lr: 0.0003624 | time: 80.303 

00101 | Train Loss: 0.0026901 | Valid Loss: 0.0805618 | lr: 0.0003624 | time: 80.303
```

**Score| Distance: 2.4882, Smoothness: 0.0756**

Result1 몸체 쪽 노드 6개 축 x, y, z -> y, z, x로 바꾸고 넣어줬음.

여태나온 결과중에 가장 효과가 좋음. 

101번까지 계속 학습이 멈추지 않는 모습을 보여서 더 길게 돌려봐도 좋을 듯.

똑같은 코드로 다시 돌려봤을 때는 valid_loss가 2.5를 돌파하지 못함.



**Run5_11**

```
00001 | Train Loss: 44939.1565925 | Valid Loss: 16.8979546 | lr: 0.0009900 | time: 79.595 

00001 | Train Loss: 0.2874453 | Valid Loss: 0.1159404 | lr: 0.0009900 | time: 79.595

00056 | Train Loss: 0.2375086 | Valid Loss: 12.9550077 | lr: 0.0005696 | time: 80.373 

00056 | Train Loss: 0.0030051 | Valid Loss: 0.0786721 | lr: 0.0005696 | time: 80.373
```

body_train_input 축 바꿈, 스케일링 함.

body_valid_input 축 바꿈, 스케일링 안함.

Result1 valid_loss 12근처에서 학습안됨.



이제 그냥 5_9버전인 축만 바꾸고 스케일링은 하지 않은 버전으로 고정해보고, 다른 과정으로 넘어가야 할 듯.

Root pos은 loss를 이용해서 잡아주자. 



일단 현재 Loss를 L1 loss로 바꿔보자



**Run6_1**

```
00001 | Train Loss: 328.8387708 | Valid Loss: 5.9363059 | lr: 0.0009900 | time: 80.224 

00001 | Train Loss: 0.1262706 | Valid Loss: 0.1010317 | lr: 0.0009900 | time: 80.224

00040 | Train Loss: 0.1669288 | Valid Loss: 0.6730136 | lr: 0.0006690 | time: 80.191 

00040 | Train Loss: 0.0175365 | Valid Loss: 0.0784421 | lr: 0.0006690 | time: 80.191
```

**Score| Distance: 1.1099, Smoothness: 0.0522 (00040기준)**

두 가지 Loss 다 L1 loss 사용해보았음.

가장 학습 잘됨. 여태 가장 효과 좋았던 Run5_9보다 훨씬 결과가 좋다.

L1 loss 사용하자.



**Run6_2**

```
00001 | Train Loss: 7.3810412 | Valid Loss: 5.3381046 | lr: 0.0009900 | time: 80.074 

00001 | Train Loss: 0.1268967 | Valid Loss: 0.0972194 | lr: 0.0009900 | time: 80.074

00011 | Train Loss: 0.1942966 | Valid Loss: 5.9898809 | lr: 0.0008953 | time: 80.091 

00011 | Train Loss: 0.0291972 | Valid Loss: 0.0797497 | lr: 0.0008953 | time: 80.091
```

**Score| Distance: 8.0734, Smoothness: 0.0365**

스케일링 적용해봤음.

valid_loss 학습이 진행이 안됨.

smoothness는 더 좋은 결과를 보임.



**Run6_3**

```
00001 | Train Loss: 237.0987700 | Valid Loss: 3.5160782 | lr: 0.0009900 | time: 79.891 

00001 | Train Loss: 0.1288813 | Valid Loss: 0.1018185 | lr: 0.0009900 | time: 79.891

00050 | Train Loss: 0.1585375 | Valid Loss: 1.1480163 | lr: 0.0006050 | time: 80.548 

00050 | Train Loss: 0.0159231 | Valid Loss: 0.0772183 | lr: 0.0006050 | time: 80.548
```

Result1 축 안바꾸고 해봄.

6_1보다 효과 안 좋음. 1밑으로 못 내려감.



**Run6_4**

```
00001 | Train Loss: 2559723.6882419 | Valid Loss: 14.6542318 | lr: 0.0009900 | time: 80.943 

00001 | Train Loss: 0.1295325 | Valid Loss: 0.1008141 | lr: 0.0009900 | time: 80.943

00090 | Train Loss: 6.2175748 | Valid Loss: 1.6907649 | lr: 0.0004047 | time: 81.515 

00090 | Train Loss: 0.0122376 | Valid Loss: 0.0750144 | lr: 0.0004047 | time: 81.515
```

**Score| Distance: 2.5497, Smoothness: 0.0757** (00090 기준)

custom loss 테스트 해볼 겸 L1 loss + MSE*0.01로 해봤음.



더 해야할 일

1. Forward Kinematic으로 좌표 구해서 loss에 넣기
2. output 갯수 9개에서 줄이기
3. Data Augumentation Noise제거 



**Run7_1**

쿼터니안 사용해서 output 갯수 9개에서 4개로 줄임.

그런데, 한 번 돌아가는 과정에서 쿼터니안 -> Rotation matrix 바꾸는 걸 2번 하다보니 너무 오래 걸림. (한 번 당 425초 정도)

label을 쿼터니안으로 바꿔서 시간을 줄여야 겠다.

또한 학습도 안됨. gradient가 제대로 안넘어가는듯



**Run7_2**

```
00001 | Train Loss: 296.4579045 | Valid Loss: 10.4428465 | lr: 0.0009900 | time: 78.480 

00001 | Train Loss: 0.1179368 | Valid Loss: 0.0728480 | lr: 0.0009900 | time: 78.480

00060 | Train Loss: 0.1175952 | Valid Loss: 0.8645066 | lr: 0.0005472 | time: 78.974 

00060 | Train Loss: 0.0141036 | Valid Loss: 0.0544812 | lr: 0.0005472 | time: 78.974
```

**Score| Distance: 1.4492, Smoothness: 0.1046**

train_label2, valid_label2을 쿼터니안으로 바꿈.

학습 진행하고 마지막 저장할 때 164->297로 바꿔줌.

Result1은 건드린 거 없는데 더 높게 나옴.

Result2은 쿼터니안 사용하니 0.7대에서 0.5대로 낮아짐.

다시 돌려봐야할듯.



**Run7_3**

```
00001 | Train Loss: 199.4441526 | Valid Loss: 4.9195067 | lr: 0.0009900 | time: 78.887 

00001 | Train Loss: 0.1185084 | Valid Loss: 0.0872796 | lr: 0.0009900 | time: 78.887

00101 | Train Loss: 0.0802696 | Valid Loss: 0.5714145 | lr: 0.0003624 | time: 78.833 

00101 | Train Loss: 0.0111648 | Valid Loss: 0.0545821 | lr: 0.0003624 | time: 78.833
```

**Score| Distance: 1.0087, Smoothness: 0.0908**

Run7_2 다시 돌려봄.

현재까지 가장 Distance 낮게 나옴. valid_loss가 더 낮아서 Smoothness도 낮게 나올줄 알았는데 6_1이 더 낮음.

쿼터니안 쓰는게 학습은 더 잘 되더라도 변환 과정이 포함되기 때문에 중간중간 큰 오류가 발생해서 점수를 깍아먹음.



**Run7_4**

Rotation mat -> 쿼터니안 다른 식으로 적용해봄.

```
00001 | Train Loss: 286.4233346 | Valid Loss: 9.6611080 | lr: 0.0009900 | time: 78.709 

00001 | Train Loss: 0.1311328 | Valid Loss: 0.0818992 | lr: 0.0009900 | time: 78.709

00090 | Train Loss: 0.0887526 | Valid Loss: 0.5185266 | lr: 0.0004047 | time: 79.094 

00090 | Train Loss: 0.0132399 | Valid Loss: 0.0611940 | lr: 0.0004047 | time: 79.094
```

**Score| Distance: 8.8858, Smoothness: 0.1223**

값이 낮아서 학습이 잘 되고 있는 줄 알았는데, 스켈레톤이 뒤집혀서 출력되었다.

쿼터니안 변환 과정에서 -해줘야 할듯.



쿼터니안에서 큰 오류 발생하는 것만 없애면 많이 좋아질 것 같다.

-> loss를 매 프레임 출력 해보자. 그리고 어느 정도 이상의 loss를 출력하면 그 전 프레임꺼 그대로 가져다 쓰기. 

-> 학습과정 중간에선 데이터를 조작하긴 힘들 것 같다. Loader에 계속 넣어줄 수 없기 때문.





 노이즈를 제거하는 방법으로

local_train_input의 각 노드가 연속된 프레임 사이에 일정 거리 이상 멀어지면 이전 프레임꺼 그대로 가져다 넣어주는 방식을 해보기로함.

그런데, 허벅지를 기준으로 스케일링 하다보니 제대로 스케일링이 안되어서 거리가 높게 측정이 될 때가 있음

키를 기준으로 스케일링 해야되나? -> 쭈구린 자세 같은 거에서 문제가 생김.





**Run7_6**

```
00001 | Train Loss: 368.7657353 | Valid Loss: 9.8470419 | lr: 0.0009900 | time: 81.036 

00001 | Train Loss: 0.2524527 | Valid Loss: 0.1151929 | lr: 0.0009900 | time: 81.036

00050 | Train Loss: 0.1577754 | Valid Loss: 0.7226207 | lr: 0.0006050 | time: 80.846 

00050 | Train Loss: 0.0256082 | Valid Loss: 0.0789595 | lr: 0.0006050 | time: 80.846
```

**Score| Distance: 1.1858, Smoothness: 0.0561**

local_train_input, local_valid_input 상대좌표들 스케일링 제대로 함. 키 22로 맞추는 식으로 스케일링함.(BVH 스케일)

쿼터니안 안쓴 버전.

큰 효과 없어 보임. 





**Run7_7**

```
00001 | Train Loss: 216.9788406 | Valid Loss: 7.4764548 | lr: 0.0009900 | time: 79.534 

00001 | Train Loss: 0.2860596 | Valid Loss: 0.0846010 | lr: 0.0009900 | time: 79.534

00020 | Train Loss: 0.5491427 | Valid Loss: 2.3251905 | lr: 0.0008179 | time: 79.358 

00020 | Train Loss: 0.0361961 | Valid Loss: 0.0535122 | lr: 0.0008179 | time: 79.358
```

쿼터니안 쓴 버전.

비슷하게 나오는듯. 



**Run7_8**

```
00001 | Train Loss: 274.8687456 | Valid Loss: 6.8227588 | lr: 0.0009900 | time: 80.685 

00001 | Train Loss: 0.2541468 | Valid Loss: 0.1245408 | lr: 0.0009900 | time: 80.685

00040 | Train Loss: 0.1649477 | Valid Loss: 0.7411015 | lr: 0.0006690 | time: 80.491 

00040 | Train Loss: 0.0291470 | Valid Loss: 0.0889525 | lr: 0.0006690 | time: 80.491
```

노이즈 제거 사용

쿼터니안 안쓴 버전

조금 더 높게 나와서 의외였다.



**Run7_9**

노이즈 제거 사용

쿼터니안 쓴 버전





Loss에 가중치 더해주고 해보기

Root pos 구하는 데이터에도 스케일링 제대로 해보기 
