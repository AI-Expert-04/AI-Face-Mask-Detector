# AI-Face-Mask-Detector

## 환경 설정

1. Window conda 환경 생성 

```
>>> conda create -n [가상환경 이름]-env python=3.6
```

1. Mac conda 환경 생성 

```
>>> conda create —name [가상환경 이름]-env python=3.8
```

2. conda 환경 활성화

```
>>> conda activate [가상환경 이름]-env
```

3. 프로젝트 경로로 이동

```
>>> cd (프로젝트 경로)
```

4. python 모듈 설치

```
>>> pip install -r requirements.txt
```  

# [데이터셋](https://api.github.com/repos/prajnasb/observations/contents/experiements/data/without_mask?ref=master)

# 얼굴 랜드마크 좌표를 이용해서 얼굴영역에 마스크 합성
# 마스크 합성된 이미지들을 train, vali로 나눔
# 마스크 안쓴 이미지들을 train, vali로 나눔
# 두개의 데이터셋으로 학습
# 얼굴 찾는 모델 가져오기. [model](https://drive.google.com/file/d/1zypxcMVbZE_KzTf5vbDQobbllZRgSwKs/view)
# 학습된 마스크 확인여부 모델, 얼굴 찾는 모델을 사용하여 최종 감지
