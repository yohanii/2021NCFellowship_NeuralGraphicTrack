

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

Normalization 적용

Normalization은 rootnp는 (104545, 32)개의 x,y,z 각각 따로 담아서 mean, std구하고 직접 적용함.

나머진 4_11과 동일

Train_loss는 많이 줄었는데, valid_loss는 5밑으로 가질 못함.

 

내 생각엔 지금 Result1은 Root position만으로 학습을 하고 있기 때문에 데이터가 너무 적어서 학습이 잘 안되는 거 같음. Root pos주변의 다른 노드들도 추가 제공하면 더 잘 될 거 같음.
