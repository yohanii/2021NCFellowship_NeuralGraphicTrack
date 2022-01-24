# Trainer
해당 프로젝트는 NC Fellowship의 Neural Graphcs Track을 위한 것으로 학습을 위해 제공됩니다.

## | Install
해당 프로젝트를 실행하기 위해서는 아래와 같은 요구 사항이 필요합니다. 사용되는 DL framework는 [Pytorch 1.9.0](https://pytorch.org/get-started/locally/)로 GPU 활용을 위해 nVidia [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive) Toolkit & [Cudnn v8.2.2](https://developer.nvidia.com/rdp/cudnn-download) 설치가 필요합니다.
GPU가 없는 경우에는 CPU 환경에서도 동작은 하지만 속도가 매우 느리므로 GPU 셋팅을 추천합니다. 초기 간단한 모델은 CPU로 가능하지만 조금만 복잡한 모델을 적용할 시엔 GPU가 필수입니다.
>제공되는 코드의 간단한 모델은 CPU에서도 동작 가능합니다.

사용 언어는 Python으로 3.6.8 버전을 권장하며 사용되는 Package는 아래와 같이 설치 가능합니다.
```
$ pip install -r requirements.txt
```
python virtual environment 사용을 권장합니다.

## | Start
전달받은 데이터셋 압축 파일을 풀고 해당 프로젝트의 dataset/ 디렉토리에 넣은 후 아래 명령어로 실행
```
$ python run_linear.py
$ python run_lstm.py
$ python run_con1d.py
```
위 코드 실행 시 사용할 subject 수 입력 (5 or all)\
5는 작은 테스트 셋으로 모델이 잘 돌아가는지 확인하는데 사용하세요.\
all는 모든 데이터셋으로 최종 학습에 사용하세요.

코드 중 "IdxValid"에 해당하는 번호는 Valid를 위한 데이터셋의 번호이며 bvh와 매칭은 dataset 폴더 내 IdxValid.txt에서 확인할 수 있습니다.


## | Code 구성
12월 2일 변경점
LSTM 및 1d CNN으로 작성된 예제 샘플이 등록되었습니다.\
이에 따라 시퀀셜하게 데이터를 추출하는 코드가 util에 추가되었습니다.

## | 중간 과제 및 평가
Network 튜닝 (~12월 17일까지 제출)

추가된 네트워크(lstm, con1d)를 참고하여 각 팀마다 자유롭게 네트워크 및 loss function을 수정하여 학습\
학습에 사용된 전체 코드와 result로 출력된 .npy(validation 번호 표기; IdxValid.txt 참조)를 압축하여 제출\
평가를 위한 모델 구동이 가능한 상태의 코드 제출 필요

평가는 비공개 Metric으로 순위를 나눌 것이며,\
평가 Metric은 모든 팀에서 중간 제출이 완료된 시점에 공개\
해당 Metric는 최종 평가에도 사용되므로 최종 평가 모델 학습 시 적극 활용하세요.

## | 최종 평가
중간 과제 종료 시 공개될 평가 Metric을 Loss 함수에 반영하여 최종 학습 모델을 만드세요.\
상세 내용은 12월 27일 릴리즈 예정
