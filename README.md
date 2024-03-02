
# 📖 Overview

뼈는 인체의 핵심 구조를 이루며, 그 정확한 분석은 의료진이 진단과 치료 계획을 수립하는 데 있어 필수적인 역할을 한다. 인공지능, 특히 딥러닝 기법을 활용한 뼈의 세밀한 분할은 의학 분야에서 점점 중요해지고 있으며, 이러한 기술은 다양한 방면에서 유용하게 활용될 수 있다. 본 프로젝트는 X-ray 이미지에서 사람의 뼈를 Segmentation 하는 모델을 만드는 것을 목표로 한다. 

<br>

## 🗂 Dataset

<img width="687" alt="1" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/f4834d1a-7ba1-4d69-a426-5d1b35964981">


<br><br>

- 데이터셋은 크게 손가락, 손등, 팔로 구성되며 총 29개의 뼈로 구성됨.
- Input : hand bone x-ray 객체가 담긴 이미지가 모델의 인풋으로 사용된다. segmentation annotation은 json file로 제공됨.
- Output : 모델은 각 클래스(29개)에 대한 확률 맵을 갖는 멀티채널 예측을 수행하고, 이를 기반으로 각 픽셀을 해당 클래스에 할당.
<br><br><br>

## 📃 Metric
- Segmentation 결과는 Test set의 Dice coefficient로 평가하며 해당 방식은 다음과 같음.
<br>

<p align="center">
  <img width="500" alt="2" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/458ddeda-41cc-41a6-aaaa-e368e6f9ce1f">
</p>




<!-- - **Annotations :** Image size, class,  -->

<!-- <br/> -->

<br><br>
# Team CV-01

## 👬🏼 Members 
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/minyun-e"><img height="110px"  src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/6ac5b0db-2f18-4e80-a571-77c0812c0bdc"></a>
            <br/>
            <a href="https://github.com/minyun-e"><strong>김민윤</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/2018007956"><img height="110px"  src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/cabba669-dda2-4ead-9f73-00128c0ae175"/></a>
            <br/>
            <a href="https://github.com/2018007956"><strong>김채아</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Eddie-JUB"><img height="110px"  src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/2829c82d-ecc8-49fd-9cb3-ae642fbe7513"/></a>
            <br/>
            <a href="https://github.com/Eddie-JUB"><strong>배종욱</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/FinalCold"><img height="110px" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/fdeb0582-a6f1-4d70-9d08-dc2f9639d7a5"/></a>
            <br />
            <a href="https://github.com/FinalCold"><strong>박찬종</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/MalMyeong"><img height="110px" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/0583f648-d097-44d9-9f05-58102434f42d"/></a>
            <br />
            <a href="https://github.com/MalMyeong"><strong>조명현</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/classaen7"><img height="110px"  src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/2806abc1-5913-4906-b44b-d8b92d7c5aa5"/></a>
              <br />
              <a href="https://github.com/classaen7"><strong>최시현</strong></a>
              <br />
          </td>
    </tr>
</table>  
      
                

</br>

## 💻 Development Environment

- GPU : Nvidia V100 32GB x 6
- 개발 언어: Python 3.10.13
- 프레임워크: Pytorch, Numpy
- 협업 툴: Github, Weight and Bias, Notion, Discord, Zoom, Google Calendar


</br>

# 📊 Project
## 🔎 EDA

<p align="center">
  <img width="350" alt="3" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/5fa475d2-fa70-4eb8-b7aa-4d925a55da36">
</p>

4/5fa475d2-fa70-4eb8-b7aa-4d925a55da36"></center>


> ### Dataset
  - Train : 800, Test : 300 (Total 1,100)
  - Resolution : 2048x2048x3
  - Class (29) : ‘finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5', 'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10', 'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15', 'finger16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium', 'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate', 'Triquetrum', 'Pisiform', 'Radius', 'Ulna'
  - 각 사람마다 두 장의 양손 이미지가 존재

> ### Noisiy Images
 다양한 종류의 노이즈가 포함된 이미지들이 발견
- 네일 아트나 매니큐어가 있는 경우
- 반지를 착용한 경우
- 수술로 인해 보형물이 삽입된 경우
- 붕대를 감은 경우 등
- 안쪽으로 회전된 손목의 데이터


<br>

## 🔬 Methods


> ### Augmentation
 
- 데이터 증강을 통한 모델 성능 향상
- 실험을 통해 다음과 같은 기법들이 모델 성능 향상에 기여함을 확인

 데이터 증강 기법 | 설명 | 효과 |
| --- | --- | --- |
| **Resize** | 이미지를 512x512 및 1024x1024로 조정 | 모델의 성능과 효율성 균형 유지 |
| **Color Jitter** | 밝기, 대비, 색조, 채도 조절 | 평균 Dice 점수 약 2% 향상 |
| **Sharpen** | 객체의 경계선 강조 | 성능 향상 (추가 파라미터 조정으로는 뚜렷한 개선 없음) |
| **Random Brightness Contrast** | 밝기와 대비 임의 조절 | 평균 Dice 점수 약 2% 증가 (Sharpen과 함께 사용 시) |
| **Gaussian Noise** | 노이즈 추가로 일반화 성능 향상 및 오버피팅 방지 | 성능 차이 미미 |
| **Elastic Transform** | 이미지에 탄성 효과 적용 | 단독 사용 시 성능 하락, Horizontal Flip과 함께 사용 시 성능 향상 |

> ### TTA
테스트 과정에서는 모델의 예측 정확도를 개선하기 위해 다음 전략을 채택
- **배치 크기 조정**: 예측 과정의 세밀함을 조절하여, 모델이 다양한 크기의 데이터를 더 효과적으로 처리할 수 있도록 함.
- **임계값 변화**: 결과의 정확도를 최적화하기 위해 예측 임계값을 조정. 세그멘테이션 작업에서의 정확도와 로버스트성을 개선하는 데 도움이 되었음


<br><br>


# 🔦 Models
- 프로젝트에서 사용된 주요 이미지 세그멘테이션 모델의 특징 및 구현 라이브러리

| Model     | Feature                                                                                     | Library                |
|------------|-----------------------------------------------------------------------------------------------|-----------------------------|
| **DeepLab v3** | Atrous convolution을 사용해 다양한 스케일에서 객체를 효과적으로 분리. ASPP 모듈로 해상도 다양성 통합. | Segmentation Models Pytorch |
| **U-Net++**   | 복잡한 네트워크 구조와 개선된 스킵 연결을 통해 세그멘테이션 정확도 향상.                         | mmsegmentation             |
| **HRNet**    | 다양한 해상도에서 이미지를 병렬로 처리하며, 해상도 간 정보 교환을 통한 세밀한 세그멘테이션 달성.    | mmsegmentation             |

<br><br>

# 📋 Conclusion

- 데이터 증강(augmentation)과 테스트-시간 증강(TTA) 방법을 통해 성능 향상을 실현
- 이전 대회와는 달리 프레임워크를 보다 효과적으로 활용하며 다양한 모델 실험을 통해 각 모델의 특성을 더 깊이 이해할 수 있었음
- loss와 관련된 실험을 진행하면서 모델 학습 시 파라미터 설정의 중요성을 배움
