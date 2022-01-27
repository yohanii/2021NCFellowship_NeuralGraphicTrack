

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

00001 | Train Loss: 0.3653895 | Valid Loss: 0.5380832 | lr: 0.0009900 | time: 3.608

00101 | Train Loss: 0.0452509 | Valid Loss: 0.4892339 | lr: 0.0003624 | time: 3.586

 

**5_CNN(MSE loss)**

\########## Start Train ##########

 00001 | Train Loss: 0.4119846 | Valid Loss: 0.5723141 | lr: 0.0009900 | time: 100.328

Model saved.

Result saved. result/subject_5_CNN/000001_result_valid

00100 | Train Loss: 0.2345810 | Valid Loss: 0.5939790 | lr: 0.0003660 | time: 100.324

Model saved.

Result saved. result/subject_5_CNN/000100_result_valid

 00101 | Train Loss: 0.2345296 | Valid Loss: 0.5887196 | lr: 0.0003624 | time: 100.166

 

 

**5_CNN(CrossEntropyloss)**

00001 | Train Loss: 0.4210672 | Valid Loss: 0.5921927 | lr: 0.0009900 | time: 11.724

00100 | Train Loss: 0.2353331 | Valid Loss: 0.5723641 | lr: 0.0003660 | time: 11.859



**5_customCNN (loss fun : CrossEntropyloss) 망한 버전.**

00001 | Train Loss: -79032341286695568.0000000 | Valid Loss: 4050498143409622.5000000 | lr: 0.0009900 | time: 19.653

Result saved. result/subject_5_CNN/000100_result_valid

 00101 | Train Loss: -11946580551261462642819072.0000000 | Valid Loss: 116507478109352062091264.0000000 | lr: 0.0003624 | time: 19.712

 

**5_customCNN (loss fun : CrossEntropyloss)**

00001 | Train Loss: 0.4041500 | Valid Loss: 0.5810984 | lr: 0.0009900 | time: 19.356

00101 | Train Loss: 0.1842482 | Valid Loss: 0.5606981 | lr: 0.0003624 | time: 19.504

Linear 보다는 성능 좋고, 기존 CNN과는 비슷

이 중에선 RNN이 가장 부드럽고 좋은 성능을 냄.

 

**5_ResNet(MSE loss)**

00001 | Train Loss: 0.5491642 | Valid Loss: 0.5714595 | lr: 0.0009900 | time: 126.209

00101 | Train Loss: 0.2076657 | Valid Loss: 0.5751043 | lr: 0.0003624 | time: 122.598

CNN모델 중에서는 가장 나은 모습.

그러나 노이즈가 엄청 심함.

 

**5_VGG(Cross Entropy loss)**

00001 | Train Loss: -38528265133739627968987136.0000000 | Valid Loss: 978813578848653325369344.0000000 | lr: 0.0009900 | time: 54.695

Result saved. result/subject_5_VGG/000100_result_valid

 00101 | Train Loss: -5606094521341963530619846656.0000000 | Valid Loss: 72911864223186672200712192.0000000 | lr: 0.0003624 | time: 54.876

망함

 

**5_VGG(MSE loss)**

00001 | Train Loss: 352.1761556 | Valid Loss: 1.6048460 | lr: 0.0009900 | time: 54.451

Result saved. result/subject_5_VGG/000100_result_valid

 00101 | Train Loss: 1.0000445 | Valid Loss: 0.7763272 | lr: 0.0003624 | time: 54.904

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

Result saved. result/subject_5_customCNN/000100_result_valid

 00101 | Train Loss: 0.1813815 | Valid Loss: 0.5848318 | lr: 0.0003624 | time: 19.500

Result saved. result/subject_5_customCNN/000100_result_valid

 00101 | Train Loss: 0.1813815 | Valid Loss: 0.5848318 | lr: 0.0003624 | time: 19.500

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

