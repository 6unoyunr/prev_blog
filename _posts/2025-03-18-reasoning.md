---
title: 추론 모델은 정말로 뛰어난가?? 오픈소스 엑사원 딥(EXAONE-Deep), QWQ, 딥시크(Deepseek-R1) 비교 실험해보기
layout: post
description: Reasoning LLMs
use_math: true
post-image: https://github.com/user-attachments/assets/f85233bb-d2ff-4554-b1e0-7e0eb7966ada
category: paper review
tags:
- Reasoning Models
- Deepseek-R1
- EXAONE
- QWQ
- o1, o3
- LLM
---

# 글이 길어질 예정이다

이 글은 추론 모델들에 대해 개인적으로 실험한 내용들을 아무런 요약이나 정리 없이 쭉 아카이빙할 목적으로 작성한다. 따라서 글이 지나치게 길어질 것으로 예상된다. 모델별 비교 결과표는 맨 아래에 첨부할 예정이니 시간이 없다면 그냥 쭉 드래그해서 맨 아래로 내리면 된다. 인터넷에 많은 사람들이 이것저것 비교해놓은 글들을 봤지만, 엑사원 딥이 새로 나온 김에 테스트해보고자 글을 작성한다.

비교 모델은 아래와 같다.

- OpenAI: o3-mini-high
- xAI: Grok3
- EXAONE-Deep:32B
- QWQ:32B
- Deepseek-R1:32B(Distillation 버전)

## 수많은 Deep 시리즈들

딥시크 추론 모델 등장 이후 너도나도 여러 가지 방법으로 학습시킨 오픈 소스 추론 모델을 내뱉기 시작한 듯하다. 엑사원 딥 모델은 LG AI Research에서 나왔는데, 문득 실제 지표처럼 잘 동작하는지 궁금해졌다. 사실 개인적으로 추론 모델이 정말 잘할 수 있는지에 대해 의문을 품고 있기 때문에, 그 호기심을 충족시키기 위해 직접 돌려보기로 했다. 일단 이것만 봐서는 EXAONE이 가장 좋긴한데, 까보기 전까지는 모르니까.

<p align="center">
    <img src="https://github.com/user-attachments/assets/afd199cc-adc4-49a5-b1e5-b34def601976" width="800">
</p>

(사진 출처는 [exaone deep 소개 페이지](https://www.lgresearch.ai/blog/view?seq=543)) 기존에는 논문 리뷰 위주로 올렸는데 테크니컬 리포트 읽는건 솔직히 노잼이라서 걍 실험해보는게 훨씬 재밌을 것 같다.

## 라이센스 비교

오픈 소스라 해서 마구잡이로 쓸 수 있는 것은 아니고 상황에 따라 다르다.

### QWQ 라이센스 요약 (Apache)

```markdown
Apache License 2.0:
- 누구나 소프트웨어를 무료로 사용, 수정, 복사, 배포, 판매, 재배포할 수 있습니다.
- 수정된 소프트웨어를 배포할 때, 원본 소스코드에 있던 저작권, 특허, 상표, 저자 표시를 그대로 유지해야 하며, 변경 사항을 명확히 표시해야 합니다.
- 이 소프트웨어에 특허가 관련된 경우, 기여자는 사용자에게 특허를 무상으로 사용할 수 있는 권리를 제공합니다. 단, 사용자가 이 소프트웨어에 대해 특허 소송을 제기하면 라이선스는 즉시 종료됩니다.
- 제공자는 소프트웨어에 대해 어떠한 보증도 하지 않으며, 사용으로 인한 손해에 대해 책임지지 않습니다(무보증 원칙).
- 배포할 때 이 라이선스를 반드시 첨부하거나 명시해야 합니다.
```

### 엑사원-딥 모델 라이센스 요약

```markdown
허용된 용도
- 비상업적 연구 목적(학술 연구, 평가, 실험 등)으로만 사용 가능
- 모델을 수정하거나 파생물을 만들 수 있으나 이름에 반드시 ‘EXAONE’을 포함해야 함
- 연구 결과는 공개 가능하며, 모델 및 파생물을 배포할 때는 반드시 이 계약서를 포함해야 함

금지된 사항
- 상업적 목적 사용(수익 창출 제품 및 서비스 개발 금지)
- 역설계, 소스코드 분석, 알고리즘 추출 금지
- 불법적, 비윤리적(허위정보 유포, 차별적 콘텐츠 제작 등) 용도 사용 금지
- 다른 AI 모델 개발 또는 개선 목적으로의 사용 금지

소유권 및 출력물
- 모델, 파생물 및 모델의 출력물(Output)에 대한 모든 권리는 Licensor(LG경영개발원)에 있음
- 출력물도 연구 목적 이외의 용도로 상업적 활용이 불가함

보증 책임 제한
- 모델은 “있는 그대로” 제공되며, Licensor는 사용 과정에서 발생하는 어떠한 손해에 대해서도 책임을 지지 않음

계약 종료
- 계약 위반 시 Licensor가 즉시 계약 종료 가능
- 계약 종료 시 즉시 모델 및 모든 복사본을 삭제해야 함

분쟁 해결
- 대한민국 법률을 따르며, 분쟁은 서울의 대한상사중재원에서 중재를 통해 해결
이 모델을 다운로드, 설치 또는 사용할 경우 위 조건에 동의하는 것으로 간주됩니다.
```

### Deepseek-R1 라이센스 요약 (MIT)

```markdown
MIT License:
- 누구나 이 소프트웨어를 무료로 자유롭게 사용, 복사, 수정, 병합, 배포, 판매할 수 있고, 다른 사람에게도 사용 권한을 줄 수 있습니다.
- 다만, 소프트웨어를 사용할 때 반드시 저작권 표시와 라이선스 원본을 포함해야 합니다.
- 소프트웨어는 어떤 형태의 보증도 없이 “있는 그대로(AS IS)” 제공되며, 개발자는 사용 과정에서 발생하는 문제나 피해에 대해 책임을 지지 않습니다.
```

일단 오픈 소스이긴 한데, 딥시크와 QWQ와는 다르게 엑사원은 상대적으로 사용에 제약이 꽤나 많다.

## 테스트한 문제들과 프롬프트 구성

1. “**How many golf balls can fit in a school bus?**" 근데 이건 따로 정답이 있는진 모르겠고 **추론 시간 비교해보려고 넣었다**.
2. **비디오 캡션 20개를 뭉텅이**로 주고, 이를 **서로 다른 3개의 paraphrased sentence로 리턴**하게함. Paraphrased 문장은 **서로 다르게 구성되어야하고 겹치는게 없어야함**.
3. **어려운 수능 수학 문제**를 하나 풀도록 해봄.
4. **어려운 코딩 문제**를 하나 풀도록 해봄.

비디오 캡션의 경우에는 MSRVTT(jsfusion 샘플 중 첫번째 샘플)에서 구했다. 해당 데이터셋을 보게 되면 비디오 데이터셋 하나당 20개의 캡션으로 구성되어있는데, 각각을 겹치지 않게 총 20(원본) + 60(phraphrased)개의 문장을  구하는 것을 목적으로 하였다.

```markdown
Paraphrase the text in [Input] according to these guidelines:
- Retain every detail from the original sentences.
- For each sentence in [Input] (each on its own line), create three unique paraphrased versions.
- In [Output], list the paraphrased sentences in the same order as the input.
- Do not include any numbering—simply output each paraphrased sentence on a new line.
- Make sure that all paraphrased sentences are different from the original sentence and from each other.

For example:
[Input]
<Original sentence 1>
<Original sentence 2>
... and so on.

[Output]
<Paraphrase version 1 of sentence 1>
<Paraphrase version 2 of sentence 1>
<Paraphrase version 3 of sentence 1>
<Paraphrase version 1 of sentence 2>
<Paraphrase version 2 of sentence 2>
<Paraphrase version 3 of sentence 2>
... and so on.

[Input]
a toy cat is bathing in soapy water in a toy bathtub
small plastic cat being washed and rubbed in a plastic tub
there is a someone playing with a deer toy
a person pretends to bathe a small plastic cat using a bathroom play set of equally small size
the head of doll is being washed by someone in soapy water
there is a hand with pink nails cleaning a kitty toy in a miniature bathtub
a person bathing a toy cat in a little bath tub
a toy cat in a toy bubble bath is being washed and groomed
a toy kitten getting a bath in a white bath tub
a person cleans a plastic toy cat in a very small bathtub
a person puts toy cat in a tub and washes it with a brush
someone cleaning a toy cat in a water tub
a  toy cat is in small toy bathtub with water and bubble and girl washes with scrubber
little pet shop cat getting a bath and washed with little brush
girl giving her adorable toy kitty a nice bubble bath
the small toys are being showered in the small bathroom
woman is cleaning the head of the orange cat
woman is putting the orange doll to have shower
woman is putting the small cat into the water and cleaning it
pony doll with big green eyes taking a bath in tiny bathtub

[Output]
```

위와 같은 프롬프트를 통해 나온 문장을 다음 코드로 분해했을때,

```python
sentences = output.split("\n") # 개행 분리
sentences = [s.strip().lower() for s in sentences if s.strip() != ""] # 소문자화
sentences = [s[:-1] if s.endswith(".") else s for s in sentences] # . 제거
print(len(sentences))
print(len(set(sentences)))
print(len(sentences + caption_list))
print(len(set(sentences + caption_list)))
```

정확하게 `60, 60, 80, 80`이 나오면 성공(정답)한 것으로 간주한다.

어려운 수능 수학 문제의 경우에는 2018년도 수학능력검정시험 30번 문제를 영어로 바꿔서 프롬프트화했으며, 문제를 프롬프트한 결과는 다음과 같다.

문제를 자세히 보니 삼각 펄스파$f(x)$와 주기 신호인 $\cos(\pi x)$와의 부분 콘볼루션인 듯하다.

<p align="center">
    <img src="https://github.com/user-attachments/assets/ff15e7f8-8d14-40c5-b99a-f0768a934483" width="400">
</p>

이걸 그대로 이미지로 넣을 순 없으니 markdown 문제 형태로 바꿔주었다.

```markdown
Question : For a real number \( t \), the function \( f(x) \) is defined as
\[
f(x) =
\begin{cases} 
1 - |x - t|, & \text{if } |x - t| \leq 1 \\ 
0, & \text{if } |x - t| > 1
\end{cases}
\]
Define the function
\[
g(t) = \int_{k}^{k+8} f(x) \cos (\pi x) \,dx.
\]
For some odd integer \( k \), the function \( g(t) \) satisfies the following condition:

The function \( g(t) \) has a local extremum at \( t = \alpha \), and for all \( \alpha \) where \( g(\alpha) < 0 \), we arrange them in ascending order as \( \alpha_1, \alpha_2, \dots, \alpha_m \) (where \( m \) is a natural number), and it is given that
\[
\sum_{i=1}^{m} \alpha_i = 45.
\]

Find the value of 
\[
k - \pi^2 \sum_{i=1}^{m} g(\alpha_i).
\]

Please reason step by step, and put your final answer within \boxed{}.
```

참고로 정답은 21이다. 풀이는 의외로 구간 잘 나누고 주기 함수의 특성만 잘 파악하면 쉽게 풀 수 있는 문제라서 가져왔다.

어려운 코딩 문제의 경우 주로 쓰이는 코딩 인터뷰 플랫폼인 리트 코드의 문제를 하나 가져오기로 했다. [420번 문제인 “strong password checker”](https://leetcode.com/problems/strong-password-checker/description/)로 정했다. 구현 형태의 문제이지만 나름 난이도가 높은 축에 속한다. 코테 문제에서는 이런 식으로 구현 문제가 많이 나오는 편이기 때문에 적절한(?) 예시이다 싶었다.

```markdown
Implement a python function `strongPasswordChecker(password: str) -> int` that returns the minimum number of steps required to make the given password strong according to the following conditions:
Code format is:

class Solution:
    def strongPasswordChecker(self, password: str) -> int:
        # Complete this function

- The password must have at least 6 characters and at most 20 characters.
- It must contain at least one lowercase letter, at least one uppercase letter, and at least one digit.
- It must not contain three repeating characters in a row (e.g., "aaa" or "111").

In one step, you may:
- Insert one character into the password,
- Delete one character from the password, or
- Replace one character of the password with another character.

If the password is already strong, return `0`.

### Examples:

strong_password_steps("a")
# Output: 5

strong_password_steps("aA1")
# Output: 3

strong_password_steps("1337C0d3")
# Output: 0

### Constraints:
- `1 <= len(password) <= 50`
- The password consists only of letters, digits, dots '.', or exclamation marks '!'.
```

## 비교할 모델들

모델을 비교할 때는 동일 파라미터 수를 기준으로 비교하는 것이 좋기 때문에 다음과 같은 모델들로 비교해보았다.

QWQ-32B, EXAONE-32B, Deepseek-R1-32B

그런데 여기에 추가로 내돈내산하고 있는 **openai의 o3-mini-high**도 사용해보았다. 추론 속도 차이에 비해 오픈 소스들 성능이 어떤지 정말 궁금해서.

그리고 최근에 나온 **Grok3**도 한번 사용해보았다. 비교 시에 따로 신경쓴 부분은 다음과 같다.

우선 각 모델은 모두 추론 모델이다. 추론 모델의 장점은 시간이 오래 걸려도 정밀한 정답을 내놓는 것인데,

사실 생각보다 **단순한 문제를 해결**할 때는 **생각을 깊게 할 필요가 없어야 하지 않을까**.

즉, 단순히 **추론 양만 늘려서 성능을 높인 케이스**라면 실제로 그닥 **효용성이 없을 수 있다는 생각**을 하게 되었다.

그러나 시간 차이가 많이 나지 않는 선에서는 추론 모델은, 어려운 문제에 대해서는 충분히 긴 시간을 생각하고 답변을 내놓는게 좋긴 할 것이다.

그래서 **개인적으로 테스트한 문제들과 이를 구성한 목적**은 다음과 같다.

1. **정답이 따로 정해져있지는 않지만 추론 과정이 중요한 문제. 과연 추론이 합리적인가?**
2. **정답이 정해져있지는 않지만 다소 간단한 문제이며, LLM이 추론 과정을 거치더라도 사용자가 원하는 output을 제대로 내어줄 수 있는가?**
3. **정답이 정해져있는 어려운 문제이며, 추론 시간에 비해 어느 정도로 정교한 답변을 내놓는가 (수학)?**
4. **정답이 정해져있는 어려운 문제이며, 추론 시간에 비해 어느 정도로 정교한 답변을 내놓는가 (코딩)?**

간단한 실험을 통해 모델의 우위를 단번에 정할 수는 없지만 **어느 정도 합리적 비교가 가능한 선에서의 문제를 select**했다. 지극히 개인적인 의견이므로 실험 결과로만 참고.

특히 수학 문제의 경우 단순히 생각하면 오답을 내놓기 쉽고 예외처리를 해야 정답이 나오는 문제를 통해 추론 모델의 파워를 가늠해보고자 하였고,

코딩 문제의 경우에도 단순히 작성하면 예외 테스트 케이스에 걸리기 쉬운 문제를 통해 추론 모델의 파워를 가늠해보고자 했다.

## 새로운 벤치마크의 필요성이 좀 느껴짐

갑자기 쓰면서 생각난 이슈인데, 추론 모델이 많이 등장할수록 스코어별로 측정하는 결과물을 복합적으로 비교해야 좀 더 fair한 비교가 되지 않을까 생각된다.

## GPU

32B 모델을 GPU에 올리기 위해서는 넉넉하게 72~75G의 메모리가 필요하다. 대용량의 서버가 있다면 한 개로도 충분한 정도이지만, 적당한 사이즈의 GPU로 돌리다보면 메모리 에러가 날 때가 많았다. GPU가 여러대인 상황에서 (코드는 huggingface 기준), 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 사용하고 싶은 모델 이름을 지정해주면 됨
model_name = "Qwen/QwQ-32B" #QWQ
# model_name = "LGAI-EXAONE/EXAONE-Deep-32B" #EXAONE
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" #Deepseek

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    cache_dir=모델 웨이트를 저장하고픈 위치(디폴트는 캐시에 저장됨)
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

위와 같이 model device를 자동으로 매핑하게 해주면 model의 safetensor가 적당히 공간이 남는 gpu에 할당이 되는 형태가 되고, Input을 넣어줄 때는

```python
output = model.generate(
        input_ids.to(model.device),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=32768,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
    )
```

위와 같이 `model.device`로 옮겨주면 된다.

CUDA에서 인식되는 gpu 중에서 **일부 gpu만 사용하고 싶을 때**는 그냥 심플하게 파이썬 스크립트를 돌릴 때 `CUDA_VISIBLE_DEVICES=` 옵션을 넣어주었다.

---

# 문제 풀이 결과들

아래는 실제로 문제를 풀이한 결과를 쭉 정리할 것인데, 너무 방대하므로 보기 귀찮은 사람들은 맨 아래쪽에 정리해둘 것이기 때문에 쭉 내리면 된다.

## 첫번째 문제 (스쿨버스 안 골프공 갯수)

<p align="center">
    <img src="https://github.com/user-attachments/assets/b8b0a264-213d-4416-897d-d3a202bad281" width="500">
</p>

### o3-mini-high

추론 시간 : 27 seconds

추론 길이 : 889

답변(think 제외) : 

**Estimating the Volume of the Bus**
Assume the interior dimensions of a typical school bus are:
- **Length:** 35 ft
- **Width:** 7.5 ft
- **Height:** 6 ft

Thus, the interior volume is: $V_{\text{bus}} = 35 \times 7.5 \times 6 \approx 1575 \text{ cubic feet}.$

Converting cubic feet to cubic inches (since $1 \text{ ft}^3 = 1728 \text{ in}^3$): $V_{\text{bus}} \approx 1575 \times 1728 \approx 2,721,600 \text{ cubic inches}.$
    
**Estimating the Volume of a Golf Ball**
    
A standard golf ball has a diameter of approximately 1.68 inches, so its radius is: $r \approx \frac{1.68}{2} = 0.84 \text{ inches}.$
The volume V of a sphere is given by: $V_{\text{ball}} = \frac{4}{3}\pi r^3 \approx \frac{4}{3}\pi (0.84)^3 \approx 2.48 \text{ cubic inches}.$
    
**Accounting for Packing Efficiency**
    
Since golf balls are spherical, they do not pack perfectly in a container. A typical packing efficiency for spheres is about 64%: 
$\text{Effective volume} \approx 0.64 \times 2,721,600 \approx 1,741,824 \text{ cubic inches}.$
    
**Calculating the Number of Golf Balls**
Dividing the effective volume of the bus by the volume of one golf ball: $\text{Number of golf balls} \approx \frac{1,741,824}{2.48} \approx 703,000.$
    
**Final Answer**
Under these approximations, roughly 700,000 golf balls could fit in a school bus.
$\boxed{700,000}$

### Grok3

추론 시간 : 98 seconds

추론 길이 : 27325

답변(think 제외) : 

To determine how many golf balls can fit in a school bus, we need to estimate the volume of a typical school bus, calculate the volume of a single golf ball, and account for the fact that spherical objects like golf balls don’t fill space perfectly due to packing efficiency. Let’s break this down step by step.

**Step 1: Estimate the School Bus Volume**

A standard full-size school bus has approximate interior dimensions that we can use for this calculation. Based on typical designs, let’s assume the interior is **40 feet long**, **8 feet wide**, and **6.5 feet high**. These dimensions account for the usable space inside, considering that the overall height of a bus might be around 10 feet, but the interior height is reduced due to the floor, ceiling, and fixtures. For simplicity, we’ll treat the bus as an empty rectangular prism, ignoring seats and other internal structures, as is common in such estimation problems.

The volume of the school bus in cubic feet is:

$V_{\text{bus}} = \text{length} \times \text{width} \times \text{height} = 40 \, \text{ft} \times 8 \, \text{ft} \times 6.5 \, \text{ft}$

Calculate:

$40 \times 8 = 320$

$320 \times 6.5 = 320 \times (6 + 0.5) = (320 \times 6) + (320 \times 0.5) = 1920 + 160 = 2080 \, \text{cubic feet}$

To work with smaller units later, convert this to cubic inches (since 1 cubic foot = 12 × 12 × 12 = 1728 cubic inches):

$V_{\text{bus}} = 2080 \times 1728$

$2000 \times 1728 = 3,456,000$

$80 \times 1728 = 138,240$

$3,456,000 + 138,240 = 3,594,240 \, \text{cubic inches}$

So, the bus has a volume of **3,594,240 cubic inches**.

**Step 2: Calculate the Golf Ball Volume**

A standard golf ball has a diameter of **1.68 inches**, so its radius is:

$r = \frac{1.68}{2} = 0.84 \, \text{inches}$

The volume of a sphere is given by:

$V_{\text{ball}} = \frac{4}{3} \pi r^3$

First, compute $r^3$:

$r = 0.84$

$0.84 \times 0.84 = 0.7056$

$0.7056 \times 0.84 \approx 0.592704$

Now calculate:

$V_{\text{ball}} = \frac{4}{3} \pi \times 0.592704$

Using $\pi \approx 3.1416$:

$\frac{4}{3} \approx 1.3333$

$1.3333 \times 3.1416 \approx 4.1888$

$4.1888 \times 0.592704 \approx 2.4827$ $\, \text{cubic inches}$

So, one golf ball occupies approximately **2.4827 cubic inches**.

**Step 3: Account for Packing Efficiency**

Golf balls are spherical, so they can’t fill the entire volume of the bus due to the gaps between them. In an ideal scenario, spheres can be arranged in a hexagonal close-packed or face-centered cubic structure, achieving a maximum packing density of:

$\rho = \frac{\pi}{3\sqrt{2}} \approx 0.7405$

This means that about 74% of the total volume can be occupied by the golf balls, with the rest being empty space. For this problem, we’ll use a packing density of **0.74**, which is a common approximation for such estimations, assuming an ordered arrangement is possible.

**Step 4: Compute the Number of Golf Balls**

The number of golf balls N N N that can fit is the bus’s volume multiplied by the packing density, divided by the volume of one golf ball:

$N = \frac{V_{\text{bus}} \times \rho}{V_{\text{ball}}}$

Substitute the values:

$N = \frac{3,594,240 \times 0.74}{2.4827}$

First, calculate the effective volume occupied by the golf balls:

$3,594,240 \times 0.74$

$3,594,240 \times 0.7 = 2,515,968$

$3,594,240 \times 0.04 = 143,769.6$

$2,515,968 + 143,769.6 = 2,659,737.6 \, \text{cubic inches}$

Now divide by the volume of one golf ball:

$N = \frac{2,659,737.6}{2.4827}$

Perform the division:

$2,659,737.6 \div 2.4827$

To estimate:

$2.4827 \times 1,071,000 \approx 2,482,700 + (2.4827 \times 71,000)$

$2.4827 \times 70,000 \approx 173,789$

$2.4827 \times 1,000 \approx 2,482.7$

$2,482,700 + 173,789 + 2,482.7 \approx 2,658,971.7$

$2,659,737.6 - 2,658,971.7 \approx 765.9$

Adjust with a small increment:

$2.4827 \times 308 \approx 764.67$

$1,071,000 + 308 = 1,071,308$

$2,658,971.7 + 764.67 \approx 2,659,736.37$

This is very close to $2,659,737.6$, confirming the calculation. Thus:

$N \approx 1,071,308$

**Step 5: Finalize the Answer**

Considering the approximations in bus dimensions and packing density, we round to a reasonable figure. The result of **1,071,308** is consistent across methods (e.g., using cubic feet yielded a similar result of 1,071,865). For simplicity and to reflect the estimation nature of the problem, we round to the nearest 10,000:

$N \approx 1,070,000$

**Conclusion**

Approximately **1,070,000 golf balls** can fit in a school bus with interior dimensions of 40 feet by 8 feet by 6.5 feet, accounting for a packing efficiency of 74%. This estimate assumes an empty bus and ideal packing, which aligns with the typical approach to such Fermi-style questions.

$\boxed{1,070,000}$

### QWQ-32B

추론 시간 : 1075.41 seconds $(\times 39.8)$

추론 길이 : 28947 $(\times 32.6)$

답변(think 제외) : 

To estimate how many golf balls can fit in a school bus, we start by calculating the volume of the school bus and the volume of a golf ball.

**School Bus Dimensions**:
- Length: 40 feet
- Width: 8 feet
- Height: 8 feet
- Total volume: $(40 \times 8 \times 8 = 2560)$ cubic feet
**Convert Bus Volume to Cubic Inches**:
- 1 cubic foot = 1728 cubic inches
- Total volume in cubic inches: $(2560 \times 1728 = 4,423,680)$ cubic inches
**Golf Ball Dimensions**:
- Diameter: 1.68 inches (radius: 0.84 inches)
- Volume of a golf ball: $(\frac{4}{3} \pi r^3 = \frac{4}{3} \pi (0.84)^3 \approx 2.48)$ cubic inches
**Calculate Number of Golf Balls**:
- Total volume divided by golf ball volume: $(\frac{4,423,680}{2.48} \approx 1,783,742)$
**Consider Packing Efficiency**:
- Spheres can be packed with a maximum density of about 74% (hexagonal close packing)
- Adjusted number: $(1,783,742 \times 0.74 \approx 1,320,000)$
**Consider Usable Space**:
- Subtract volume occupied by seats and other structures. Assuming seats take up about 630 cubic feet (1,088,640 cubic inches):
- Usable volume: $(4,423,680 - 1,088,640 = 3,335,040)$ cubic inches
- Adjusted number: $(\frac{3,335,040}{2.48} \times 0.74 \approx 1,000,000)$

After considering the packing efficiency and the usable space within the school bus, the final estimate is approximately one million golf balls.

$\boxed{1,000,000}$

### EXAONE-32B

추론 시간 : 881.52 seconds $(\times32.6)$

추론 길이 : 23396 $(\times 26.3)$

답변(think 제외) : 

To determine how many golf balls can fit in a school bus, we start by estimating the dimensions of a typical school bus and the volume of a golf ball.

**Assumptions and Calculations:**

**School Bus Dimensions:**
- Length: 35 feet (420 inches)
- Width: 8 feet (96 inches)
- Height: 9.5 feet (114 inches)
- Volume: $420 \, \text{in} \times 96 \, \text{in} \times 114 \, \text{in} = 4,596,480 \, \text{cubic inches}$
**Golf Ball Dimensions:**
- Diameter: 1.68 inches (radius = 0.84 inches)
- Volume of one golf ball: $\frac{4}{3} \pi (0.84)^3 \approx 2.483 \, \text{cubic inches}$
**Packing Efficiency:**
- For random packing of spheres, the efficiency is approximately 64% (0.64).
**Total Number of Golf Balls:**
- Calculate the effective volume occupied by golf balls considering packing efficiency:
$\text{Total Volume} \times \text{Packing Efficiency} = 4,596,480 \, \text{in}^3 \times 0.64 = 2,941,755.2 \, \text{in}^3$
- Divide by the volume of one golf ball:
$\frac{2,941,755.2 \, \text{in}^3}{2.483 \, \text{in}^3/\text{golf ball}} \approx 1,184,758$

Rounding to a reasonable significant figure, the approximate number of golf balls that can fit in a school bus is:

$\boxed{1,200,000}$

### Deepseek-R1-32B

추론 시간 : 317.70 seconds ($\times 11.8$)

추론 길이 : 8487

답변(think 제외) : 

Approximately 500,000 to 1,000,000 golf balls can fit in a school bus, with a more precise estimate around 700,000 when accounting for packing density.

**Step-by-Step Explanation:**

**Determine the dimensions of a school bus:**
- Length: 30 feet (360 inches)
- Width: 8 feet (96 inches)
- Height: 6 feet (72 inches)
**Calculate the volume of the school bus in cubic feet:**
- Volume = 30 ft × 8 ft × 6 ft = 1,440 cubic feet
**Convert the bus volume to cubic inches:**
- 1 cubic foot = 1,728 cubic inches
- Bus volume = 1,440 × 1,728 = 2,488,320 cubic inches
**Calculate the volume of a golf ball:**
- Diameter = 1.68 inches → Radius = 0.84 inches
- Volume = (4/3)πr³ ≈ 2.48 cubic inches
**Determine the number of golf balls without considering packing efficiency:**
- Number = 2,488,320 ÷ 2.48 ≈ 1,003,355 golf balls
**Adjust for packing efficiency (approximately 74% for spheres):**
- Number ≈ 1,003,355 × 0.74 ≈ 742,488 golf balls
**Consider practical limitations (e.g., seats, structure):**
- The estimate is adjusted to a range of 500,000 to 1,000,000 golf balls, with a more precise figure around 700,000.

**Answer:** Approximately 500,000 to 1,000,000 golf balls can fit in a school bus, with a more precise estimate around 700,000 considering packing density and practical limitations.

$\boxed{700,000}$

## 두번째 문제 (Caption paraphrasing)

<p align="center">
    <img src="https://github.com/user-attachments/assets/1402206e-c47a-45ff-b83a-a96212d1444a" width="500">
</p>

### o3-mini-high

추론 시간 : 21 seconds

추론 길이 : 0 (N/A)

답변 : `60(list), 60(set), 80(list), 80(set)`

```markdown
A toy cat is immersed in a soapy bath inside a miniature bathtub.
Inside a small tub filled with soapy water, a toy cat is taking a bath.
A miniature bathtub holds soapy water while a toy cat enjoys a bath.

A small plastic cat is getting washed and scrubbed in a plastic tub.
In a plastic tub, a tiny plastic cat is being cleaned and rubbed.
A diminutive plastic cat undergoes washing and rubbing in a plastic container.

Someone is playing with a toy deer.
A person is engaging in play with a deer toy.
There is an individual interacting with a deer toy.

A person feigns bathing a tiny plastic cat with a miniature bathroom playset.
Using a small-scale bathroom play set, an individual pretends to wash a small plastic cat.
An individual simulates giving a bath to a diminutive plastic cat with an equally small bathroom play set.

Someone is washing the doll’s head in a basin of soapy water.
The doll’s head is being cleaned with soapy water by an individual.
An individual cleans the head of the doll using water mixed with soap.

A hand with pink nails is cleaning a kitty toy inside a miniature bathtub.
Inside a tiny bathtub, a hand sporting pink nails is washing a toy kitty.
A pink-nail adorned hand is seen cleaning a kitty toy in a small bathtub.

A person is bathing a toy cat in a small bathtub.
In a little tub, an individual is washing a toy cat.
An individual gives a bath to a toy cat in a petite bathtub.

A toy cat enjoying a bubble bath is being washed and groomed.
In a playful bubble bath, a toy cat is receiving washing and grooming.
A toy cat, immersed in a bubble bath, is undergoing both cleaning and grooming.

A toy kitten is taking a bath in a white bathtub.
In a white tub, a toy kitten is being bathed.
A toy kitten is having its bath in a pristine white bathtub.

An individual cleans a plastic toy cat in an extremely small bathtub.
In a very tiny bathtub, a person washes a plastic toy cat.
A plastic toy cat is being cleaned by someone in a very diminutive tub.

A person places a toy cat into a tub and washes it with a brush.
An individual sets the toy cat in a tub and cleans it using a brush.
The toy cat is put into a tub and scrubbed with a brush by a person.

Someone is washing a toy cat in a water-filled tub.
An individual is cleaning a toy cat inside a tub of water.
A toy cat is being scrubbed in a tub filled with water by someone.

A toy cat sits in a small toy bathtub filled with water and bubbles while a girl cleans it with a scrubber.
In a little bathtub with water and bubbles, a toy cat is being washed by a girl using a scrubber.
A girl is scrubbing a toy cat placed in a small bathtub containing water and bubbles.

A small pet shop cat is receiving a bath and being washed with a tiny brush.
A little pet shop cat is being bathed and cleaned using a small brush.
An adorable pet shop cat is undergoing a bath, washed with a little brush.

A girl is providing her charming toy kitty with a pleasant bubble bath.
The girl treats her adorable toy kitty to a refreshing bubble bath.
Her delightful toy kitty is receiving a lovely bubble bath from a girl.

In a compact bathroom, the small toys are getting showered.
The tiny toys are receiving a shower in a modest bathroom.
In a little bathroom, small toys are being bathed with a shower.

A woman is washing the head of an orange cat.
The head of the orange cat is being cleaned by a woman.
A woman is tidying up the head of the orange-colored cat.

A woman is placing the orange doll in a shower.
The orange doll is being set up for a shower by a woman.
A woman is preparing the orange doll for a bath in the shower.

A woman is placing the small cat into water and washing it.
The small cat is being immersed in water and cleaned by a woman.
A woman puts the tiny cat into water and gives it a thorough cleaning.

A pony doll featuring large green eyes is bathing in a tiny bathtub.
The pony doll with big green eyes is taking a bath in a small tub.
A doll in the shape of a pony with prominent green eyes is having a bath in a diminutive bathtub.
```

### Grok3

추론 시간 : 93 seconds

추론 길이 : 36220

답변 : `60(list), 60(set), 80(list), 80(set)`

```markdown
A toy cat is taking a bath in a toy bathtub filled with soapy water.  
Soapy water fills a toy bathtub where a toy cat is bathing.  
Bathing in soapy water, a toy cat is in a toy bathtub.  

A small cat made of plastic is being washed and rubbed in a tub also made of plastic.  
In a plastic tub, someone is washing and rubbing a small plastic cat.  
Washing and rubbing are being applied to a small plastic cat inside a plastic tub.  

A person is engaging in play with a toy deer.  
There is an individual interacting playfully with a deer figurine.  
Someone is having fun with a toy that resembles a deer.  

Using a miniature bathroom play set, a person is simulating the bathing of a small plastic cat.  
A small plastic cat is being pretend-bathed by a person with the help of an equally tiny bathroom play set.  
In a make-believe scenario, a person uses a small bathroom play set to bathe a plastic cat of similar size.  

Someone is cleaning the head of a doll using soapy water.  
In soapy water, the head of a doll is being washed by a person.  
A person is washing the doll’s head with soapy water.  

A hand, notable for its pink nails, is washing a toy kitten in a small bathtub.  
In a tiny bathtub, a kitty toy is being cleaned by a hand that has pink nails.  
Cleaning a kitty toy in a miniature bathtub is a hand that has pink nails.  

An individual is giving a bath to a toy cat in a small bathtub.  
In a tiny bathtub, a person is bathing a toy cat.  
A toy cat is being bathed by a person using a little bathtub.  

A toy cat is getting washed and groomed while in a bubble bath designed for toys.  
In a toy-sized bubble bath, a toy cat is being cleaned and cared for.  
Washing and grooming are being performed on a toy cat that is in a toy bubble bath.  

A toy kitten is being bathed in a bathtub that is white.  
In a white-colored bathtub, a toy kitten is receiving a bath.  
A bath is being given to a toy kitten inside a white bathtub.  

Using a tiny bathtub, a person is cleaning a toy cat made of plastic.  
A plastic toy cat is being cleaned by a person in a bathtub of very small size.  
In a minuscule bathtub, a person is washing a plastic cat toy.  

A person places a toy cat into a tub and proceeds to wash it using a brush.  
Using a brush, a person washes a toy cat after putting it in a tub.  
A toy cat is placed in a tub by a person who then washes it with a brush.  

A person is washing a toy cat inside a tub filled with water.  
In a tub containing water, someone is cleaning a toy cat.  
A toy cat is being cleaned by someone using a water-filled tub.  

A girl is using a scrubber to wash a toy cat that is in a small toy bathtub filled with water and bubbles.  
In a tiny toy bathtub containing water and bubbles, a toy cat is being washed by a girl with a scrubber.  
A toy cat, placed in a small bathtub with bubbly water, is being cleaned by a girl using a scrubber.  

A toy cat from the Little Pet Shop is being given a bath and cleaned with a small brush.  
Using a tiny brush, someone is washing a Little Pet Shop cat toy during its bath.  
A bath is being given to a Little Pet Shop cat toy, and it is being washed with a little brush.  

A girl is providing a pleasant bubble bath to her cute toy kitten.  
Her charming toy kitty is receiving a lovely bubble bath from a girl.  
A nice bubble bath is being given by a girl to her adorable toy cat.  

In a miniature bathroom setting, small toys are being washed as if in a shower.  
The small toys are getting a shower in a tiny bathroom designed for them.  
A shower is being simulated for the small toys within a small-scale bathroom.  

A woman is washing the head of a cat that is orange.  
The head of an orange cat is being cleaned by a woman.  
An orange cat’s head is getting cleaned by a woman.  

A woman is positioning the orange doll so that it can take a shower.  
The orange doll is being placed by a woman to receive a shower.  
To give it a shower, a woman is setting the orange doll in place.  

A woman is placing the small cat into water and proceeding to clean it.  
Into the water, a woman is putting the small cat and then cleaning it.  
The small cat is being put into water by a woman who is also cleaning it.  

A pony doll that has large green eyes is bathing in a small bathtub.  
In a miniature bathtub, a pony doll with big green eyes is taking a bath.  
Taking a bath in a tiny bathtub is a pony doll characterized by its big green eyes.
```

### QWQ-32B

추론 시간 : 180.73 seconds$(\times 8.6)$

추론 길이 : 6498

답변 : `60(list), 60(set), 80(list), 80(set)`

```markdown
a plastic feline toy is submerged in sudsy water within a miniature bathtub
The cat-shaped plaything is immersed in bubbly liquid inside a small-scale bath basin.
A miniature cat toy is being cleansed with soapy solution in a scaled-down bathroom set.

A diminutive plastic feline is cleansed and massaged in a tiny plastic container.
The little plastic cat toy is scrubbed and stroked while submerged in a small plastic basin.
A small plastic cat figure is undergoing cleaning and gentle rubbing inside a miniature plastic tub.

An individual is interacting with a deer-shaped play object.
A person is engaging with a toy deer during playtime.
Someone is manipulating a deer-themed toy in a playful manner.

An individual acts out washing a tiny plastic cat with a miniature bathroom playset.
Someone simulates bathing a small plastic feline using a scaled-down bathroom toy set.
A person role-plays cleaning a diminutive plastic cat with a toy bathroom ensemble of matching size.

The doll's head is being cleansed with soapy solution by an individual.
Someone is washing the head portion of a doll using sudsy water.
A person is scrubbing the upper part of a doll's body in soapy liquid.

A hand adorned with pink nails is scrubbing a kitten toy in a tiny bathtub.
The fingers with pink nail polish are cleaning a small cat figurine in a miniature bath.
A manicured hand with pink-colored nails tends to a kitty toy in a diminutive tub.

An individual is rinsing a feline toy in a diminutive bathtub setup.
A person is submerging a toy cat in a small bath tub for cleaning.
Someone bathes a cat-shaped plaything using a little plastic tub.

A plastic cat toy is in a bubble-filled bath and is being cleaned thoroughly.
The toy cat is being washed and groomed while surrounded by suds in a tiny tub.
A miniature cat figurine is undergoing a sudsy cleansing in a bubble-filled basin.

A toy kitten is receiving a bath in a white-colored bathtub.
The small cat toy is immersed in soapy water within a white tub.
A kitten-shaped plaything is getting cleansed in a pristine white bathtub.

A woman is scrubbing a plastic cat toy in an extremely small bathtub.
An individual cleans a tiny plastic feline in a bath tub of minimal size.
A person is washing a miniature plastic cat in a very petite bathtub.

A person places a toy cat into a tub and scrubs it with a brush.
Someone positions a cat-shaped toy in a bath tub and cleans it using a brush.
An individual is using a brush to wash a feline toy submerged in a small tub.

Someone is cleaning a cat toy in a water-filled tub.
A person is scrubbing a toy cat in a tub containing water.
An individual is washing a feline plaything in a water-filled basin.

A miniature cat toy sits in a small tub filled with water and bubbles, being scrubbed by a girl.
The toy cat is in a tiny bathtub with water and foam, being washed with a scrubber by a child.
A girl is using a scrubber to clean a cat toy in a miniature tub with bubbles and water.

A petite Pet Shop cat toy is being bathed with a small brush.
The little cat from Pet Shop is getting washed using a tiny brush.
A miniature Pet Shop feline toy is scrubbed with a diminutive brush during bath time.

A young girl is giving her fluffy toy cat a sudsy bath.
The child is providing her soft toy kitten with a bubbly cleansing.
A girl is bathing her adorable kitten toy in foamy water.

Small toys are being rinsed under a stream of water in a tiny bathroom.
Miniature playthings are being hosed down in a scaled-down bathroom area.
The little toys are undergoing a shower in a diminutive bathroom setup.

A woman is washing the head of an orange-colored cat toy.
The lady is scrubbing the head section of an orange-hued doll.
A woman cleans the upper part of an orange cat figure with water.

A woman is assisting an orange doll take a shower.
The female is positioning the orange doll under a showerhead for cleansing.
A woman helps the orange-colored doll get a shower.

A woman submerges a small cat toy into water and cleans it meticulously.
The lady is immersing the tiny cat figure in water and scrubbing it thoroughly.
A woman places a miniature cat into water and scrubs it clean.

A doll resembling a pony with large green eyes is in a tiny bathtub.
The pony-shaped doll, featuring big green eyes, is taking a bath in a small tub.
A pony doll with oversized green eyes is submerged in a miniature bathtub.
```

### EXAONE-32B

추론 시간 : 211.85 seconds $(\times10.1)$

추론 길이 : 9243

답변 : `33(list), 33(set), 53(list), 53(set)`

```markdown
a miniature tub holds a plastic cat soaking in bubbly liquid
someone's scrubbing a small feline toy in a child-sized bathroom setup
in a toy sink, a feline figurine is being washed with soapy water

a plastic feline is being scrubbed and massaged in a tub
in a small tub, a toy cat's surface is being cleaned with soapy water
someone is bathing a tiny cat model in a bathtub using a sponge

a person interacts with a deer-shaped toy while laughing
a deer figurine is being amused by a child
a youngster is entertaining a stuffed deer

a child uses a miniature bathroom set to simulate washing a tiny cat figurine
someone is role-playing a cat bath scenario with a scaled-down bathroom accessory
in a play scenario, an adult imitates cleaning a small cat model with a tiny sink setup

a soapy solution is used to clean a doll's scalp by a caregiver
someone is rinsing the top part of a plastic doll in bubbly liquid
a person applies soap to a toy doll's forehead and rinses it off

a hand with pink nails wiping a kitty toy in a mini tub
the feline toy's fur is being lathered in a plastic basin by a child
an individual dries a small cat model after soaking in a compact tub

a person massages a toy feline in a compact bathroom playmat
a kiddie tub holds a plastic cat being squirted with soapy water
someone is pretending to groom a small cat in a scaled-down bathroom

a toddler giggles as they "wash" a plush cat in a toy sink
a caretaker suds up a miniature cat in a tiny tub using dish soap
in a dollhouse bathroom, a figurine cat is being scrubbed with a toothbrush

a girl uses bubble bath to cleanse a kitty figurine in a palm-sized tub
a child's play set includes a cat getting hosed in a miniature shower
a tiny tub's occupant, a plastic cat, is being patted dry with a cloth

a parent demonstrates washing a toy cat in a child-sized tub for their kid
the orange doll's fur is soaked in a small basin by a woman
a woman simulates bathing an orange-colored toy cat in a mini tub

a woman squeezes soap onto a pony figurine's coat in a teeny tub
a pony toy with large green eyes is half-submerged in a shallow tub
someone is dunking a horse-themed plaything in a tiny basin for cleaning
```

### Deepseek-R1-32B

추론 시간 : 152.59 seconds ($\times 7.3$)

추론 길이 : 5874

답변 : `60(list), 60(set), 80(list), 79(set)`

```markdown
A toy cat is taking a bath in soapy water inside a toy bathtub.
In a toy bathtub, a toy cat is being washed with soap.
A toy cat is submerged in soapy water within a toy bathtub.

A small plastic cat is being cleaned and scrubbed in a plastic tub.
In a plastic tub, a tiny plastic cat is being washed and rubbed.
A plastic cat is being scrubbed and rinsed in a plastic container.

Someone is playing with a deer-shaped toy.
A person is interacting with a deer toy.
There is an individual engaging with a toy deer.

A person is pretending to bathe a small plastic cat using a miniature bathroom set.
Using a small bathroom playset, a person is bathing a tiny plastic cat.
A person is acting out a bathing scenario with a small plastic cat and a tiny bathroom set.

The head of a doll is being cleaned in soapy water by someone.
Someone is washing the doll's head with soap and water.
A person is scrubbing the doll's head in soapy water.

A hand with pink nails is cleaning a kitty toy in a tiny bathtub.
In a miniature bathtub, a hand with pink nails is washing a cat toy.
A hand adorned with pink nails is scrubbing a toy kitten in a small tub.

A person is bathing a toy cat in a little bathtub.
In a small bathtub, a person is washing a toy cat.
A person is giving a toy cat a bath in a tiny tub.

A toy cat in a toy bubble bath is being washed and groomed.
In a toy bubble bath, a toy cat is being cleaned and brushed.
A toy cat is being bathed and groomed in a bubble-filled toy tub.

A toy kitten is getting a bath in a white bathtub.
In a white bathtub, a toy kitten is being washed.
A toy kitten is submerged in water in a white tub.

A person is cleaning a plastic toy cat in a very small bathtub.
In a tiny bathtub, a person is scrubbing a plastic toy cat.
A plastic toy cat is being washed in a very small bathtub by someone.

A person is placing a toy cat in a tub and washing it with a brush.
In a tub, a person is washing a toy cat with a brush.
A person is scrubbing a toy cat in a tub using a brush.

Someone is cleaning a toy cat in a water-filled tub.
In a water tub, someone is washing a toy cat.
A person is bathing a toy cat in a tub filled with water.

A toy cat is in a small toy bathtub with water, bubbles, and a girl washing it with a scrubber.
In a small toy bathtub, a toy cat is being washed by a girl with a scrubber.
A girl is scrubbing a toy cat in a small tub filled with water and bubbles.

A little pet shop cat is getting a bath and being washed with a small brush.
In a small tub, a pet shop cat is being bathed with a tiny brush.
A little cat from a pet shop is being washed and groomed with a small brush.

A girl is giving her adorable toy kitty a nice bubble bath.
In a bubble bath, a girl is washing her cute toy kitten.
A girl is providing her adorable toy cat with a luxurious bubble bath.

Small toys are being showered in a small bathroom.
In a tiny bathroom, small toys are being cleaned under water.
Small toys are being rinsed in a miniature bathroom setting.

A woman is cleaning the head of an orange cat.
An orange cat's head is being washed by a woman.
A woman is scrubbing the head of an orange cat.

A woman is putting an orange doll in the shower.
An orange doll is being placed in the shower by a woman.
A woman is giving an orange doll a shower.

A woman is putting a small cat into the water and cleaning it.
A small cat is being placed in water and washed by a woman.
A woman is submerging a small cat in water and cleaning it.

A pony doll with big green eyes is taking a bath in a tiny bathtub.
In a tiny bathtub, a pony doll with large green eyes is being bathed.
A pony doll with big green eyes is submerged in water in a small tub.
```

## 세번째 문제 (2018년도 수능 수학 문제 30번)

<p align="center">
    <img src="https://github.com/user-attachments/assets/1f3a726d-5f2a-4f6c-8c69-f1bb9952b61d" width="500">
</p>

### o3-mini-high

추론 시간 : 55 seconds

추론 길이 : 1413

답변(think 제외) : `오답`  
답변요약 : 적분 구간에 대한 조건화 실패  
**Final Answer** : $\boxed{23}.$

### Grok3

추론 시간 : 97 seconds

추론 길이 : 30894

답변(think 제외) : `오답`  
답변요약 : 적분 구간에 대한 조건화 실패  
**Final Answer** : $\boxed{23}.$

### EXAONE-32B

추론 시간 : 1265.00 seconds $(\times23.0)$

추론 길이 : 30510 $(\times 21.6)$

답변(think 제외) : `오답`  
답변요약 : 적분 구간에 대한 조건화 실패  
**Final Answer** : $\boxed{23}.$

### Deepseek-R1-32B

추론 시간 : 2109.8 seconds ($\times 38.4$)

추론 길이 : 46070 ($\times 32.6$)

답변(think 제외) : `오답`  
답변요약 : 걍 틀림  
**Final Answer** : $\boxed{50}.$

## 네번째 문제 (리트코드)

<p align="center">
    <img src="https://github.com/user-attachments/assets/9b202087-46b2-421c-b610-b0140768834d" width="500">
</p>

### o3-mini-high

추론 시간 : 57 seconds

추론 길이 : 1625

답변(코드만) : `정답`

```python
class Solution:
    def strongPasswordChecker(self, password: str) -> int:
        n = len(password)
        
        # Check for missing types: lowercase, uppercase, digit
        missing_types = int(not any(c.islower() for c in password)) \
                        + int(not any(c.isupper() for c in password)) \
                        + int(not any(c.isdigit() for c in password))
        
        # Find repeating sequences of 3 or more characters
        repeats = []
        i = 0
        while i < n:
            j = i
            while j < n and password[j] == password[i]:
                j += 1
            run_len = j - i
            if run_len >= 3:
                repeats.append(run_len)
            i = j

        # Case 1: Password too short: insertions needed.
        if n < 6:
            return max(missing_types, 6 - n)
        
        # Case 2: Password length is acceptable.
        if n <= 20:
            # Each repeating sequence of length L needs L//3 replacements.
            replace = 0
            for run_len in repeats:
                replace += run_len // 3
            return max(missing_types, replace)
        
        # Case 3: Password too long: need deletions.
        delete = n - 20
        
        # Optimize deletions by targeting repeating sequences:
        # First, we try to reduce replacements by deleting in groups with (L % 3 == 0)
        rep_list = repeats[:]  # copy the list so we can modify lengths
        for i in range(len(rep_list)):
            if delete <= 0:
                break
            if rep_list[i] < 3:
                continue
            if rep_list[i] % 3 == 0:
                # Remove one character from this group to reduce one replacement.
                rep_list[i] -= 1
                delete -= 1
        
        # Then, for groups with L % 3 == 1, deleting 2 characters reduces a replacement.
        for i in range(len(rep_list)):
            if delete <= 0:
                break
            if rep_list[i] < 3 or rep_list[i] % 3 != 1:
                continue
            # How many deletions can we perform? We delete in chunks of 2.
            needed = min(delete, 2)
            rep_list[i] -= needed
            delete -= needed
        
        # Finally, for any remaining groups, remove as many as possible.
        for i in range(len(rep_list)):
            if delete <= 0:
                break
            if rep_list[i] < 3:
                continue
            # Remove as many characters as possible; each 3 deleted reduces one replacement.
            needed = rep_list[i] - 2
            remove = min(delete, needed)
            rep_list[i] -= remove
            delete -= remove
        
        # After deletions, calculate how many replacements are needed for the repeats.
        replace = 0
        for run_len in rep_list:
            if run_len >= 3:
                replace += run_len // 3
        
        # Total steps = deletions performed + maximum of missing types and replacements.
        return (n - 20) + max(missing_types, replace)
```

### Grok3

추론 시간 : 380 seconds

추론 길이 : 118329

답변(코드만) : `오답(36/54)` 

```python
class Solution:
    def strongPasswordChecker(self, password: str) -> int:
        # Get the length of the password
        n = len(password)
        
        # Calculate missing character types
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        missing = 3 - (has_lower + has_upper + has_digit)
        
        # Calculate operations needed for repeating sequences
        repeat_ops = 0
        i = 0
        while i < n:
            j = i
            # Count length of current sequence
            while j < n and password[j] == password[i]:
                j += 1
            count = j - i
            if count >= 3:
                repeat_ops += count // 3  # Integer division for floor(k / 3)
            i = j
        
        # Determine minimum steps based on length
        if n < 6:
            return max(6 - n, missing, repeat_ops)
        elif 6 <= n <= 20:
            return max(missing, repeat_ops)
        else:  # n > 20
            return max(n - 20, missing, repeat_ops)
```

### QWQ-32B

추론 시간 :  2053.40 seconds$(\times 36.0)$

추론 길이 : 65257 $(\times 40.2)$

답변(코드만) : `오답(28/54)`

```python
class Solution:
    def strongPasswordChecker(self, password: str) -> int:
        n = len(password)
        if n == 0:
            return 6  # Need at least 6 characters with required types

        # Check for missing character types
        has_lower = has_upper = has_digit = False
        for c in password:
            if c.islower():
                has_lower = True
            elif c.isupper():
                has_upper = True
            elif c.isdigit():
                has_digit = True

        missing_types = 0
        if not has_lower:
            missing_types += 1
        if not has_upper:
            missing_types += 1
        if not has_digit:
            missing_types += 1

        # Compute runs of repeating characters
        runs = []
        if n == 0:
            runs = []
        else:
            current_char = password[0]
            current_length = 1
            for c in password[1:]:
                if c == current_char:
                    current_length += 1
                else:
                    runs.append(current_length)
                    current_char = c
                    current_length = 1
            runs.append(current_length)

        total_repeats = sum(r // 3 for r in runs)

        if n < 6:
            return max(6 - n, missing_types + total_repeats)
        elif n <= 20:
            return max(missing_types, total_repeats)
        else:
            surplus = n - 20

            def calculate_reduction(runs, s):
                mod0 = []
                mod1 = []
                mod2 = []
                for r in runs:
                    m = r % 3
                    if m == 0:
                        mod0.append(r)
                    elif m == 1:
                        mod1.append(r)
                    else:
                        mod2.append(r)

                reduction = 0
                s_copy = s

                # Process mod0 runs first
                for r in mod0:
                    possible = r // 3
                    take = min(possible, s_copy)
                    reduction += take
                    s_copy -= take
                    if s_copy == 0:
                        break

                if s_copy == 0:
                    return reduction

                # Process mod1 runs next
                for r in mod1:
                    possible = r // 3
                    take = min(possible, s_copy // 2)
                    reduction += take
                    s_copy -= take * 2
                    if s_copy < 0:
                        # Adjust for overstepping
                        take = (s_copy + take * 2) // 2
                        reduction += take
                        s_copy = 0
                        break
                    if s_copy == 0:
                        break

                if s_copy == 0:
                    return reduction

                # Process mod2 runs last
                for r in mod2:
                    possible = r // 3
                    take = min(possible, s_copy // 3)
                    reduction += take
                    s_copy -= take * 3
                    if s_copy < 0:
                        take = (s_copy + take * 3) // 3
                        reduction += take
                        s_copy = 0
                        break
                    if s_copy == 0:
                        break

                return reduction

            reduction = calculate_reduction(runs, surplus)
            adjusted_repeats = max(0, total_repeats - reduction)
            return max(surplus, adjusted_repeats + missing_types)
```

### EXAONE-32B

추론 시간 : 1952.09 seconds $(\times34.2)$

추론 길이 : 57913 $(\times 35.6)$

답변(코드만) : `오답(51/54)`

```python
class Solution:
    def strongPasswordChecker(self, password: str) -> int:
        # Check for missing character types
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        missing_types = 3 - (has_lower + has_upper + has_digit)

        # Calculate runs of repeating characters
        runs = []
        n = len(password)
        if n == 0:
            runs = []
        else:
            current_char = password[0]
            count = 1
            for c in password[1:]:
                if c == current_char:
                    count += 1
                else:
                    runs.append(count)
                    current_char = c
                    count = 1
            runs.append(count)

        required_repeats = sum(L // 3 for L in runs)

        if n < 6:
            insert_needed = 6 - n
            return max(insert_needed, missing_types, required_repeats)
        elif 6 <= n <= 20:
            return max(missing_types, required_repeats)
        else:
            # Case when length exceeds 20
            delete_needed = n - 20
            mod0, mod1, mod2 = [], [], []
            for L in runs:
                mod = L % 3
                if mod == 0:
                    mod0.append(L)
                elif mod == 1:
                    mod1.append(L)
                else:
                    mod2.append(L)

            adjusted_repeats = required_repeats
            del_remaining = delete_needed

            # Process mod0 first (each can take 1 deletion to reduce repeats by 1)
            use_mod0 = min(len(mod0), del_remaining)
            adjusted_repeats -= use_mod0
            del_remaining -= use_mod0

            # Process mod1 (each 2 deletions reduce repeats by 1 per run)
            if del_remaining > 0:
                possible_pairs = del_remaining // 2
                use_pairs = min(possible_pairs, len(mod1))
                adjusted_repeats -= use_pairs
                del_remaining -= 2 * use_pairs

            # Process mod2 (each 3 deletions reduce repeats by 1)
            if del_remaining > 0:
                possible_triple = del_remaining // 3
                adjusted_repeats -= possible_triple
                del_remaining -= 3 * possible_triple

            # Total steps is deletions plus max of adjusted_repeats and missing_types
            return delete_needed + max(adjusted_repeats, missing_types)
```

### Deepseek-R1-32B

추론 시간 : 1152.13 seconds ($\times 20.2$)

추론 길이 : 41756 ($\times 25.7$)

답변(코드만) : `오답(24/54)`

```python
class Solution:
    def strongPasswordChecker(self, password: str) -> int:
        # Compute missing types
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        missing = 0
        if not has_lower:
            missing += 1
        if not has_upper:
            missing += 1
        if not has_digit:
            missing += 1

        # Compute runs of 3 or more identical characters
        runs = []
        if not password:
            return 0  # Though constraints say len >=1
        current = password[0]
        count = 1
        for c in password[1:]:
            if c == current:
                count += 1
            else:
                if count >= 3:
                    runs.append(count)
                current = c
                count = 1
        if count >= 3:
            runs.append(count)

        # Calculate the steps needed for repeating sequences
        sum_repeats_steps = sum((r - 1) // 2 for r in runs)

        n = len(password)
        if n < 6:
            return max(6 - n, missing + sum_repeats_steps)
        elif n <= 20:
            return max(missing, sum_repeats_steps)
        else:
            del_num = n - 20
            return del_num + sum_repeats_steps
```

---

# 결과 정리

## 1번 문제 (스쿨버스 안 골프공 갯수)

| **Model** | **정답 여부(정답 없음)** | **처리 시간** | **추론 길이** | **단위 시간당 추론(char/sec)** | **Rank** |
| --- | --- | --- | --- | --- | --- | --- |
| o3-mini-high | 700000개 | 27초 | 880 | 32.6 | 1 |
| Grok3 | 1070000개 | 98초 | 27325 | 278.8 | 4 |
| QWQ:32B | 1000000개 | 1075.41초 | 28947 | 26.9 | 5 |
| EXAONE-Deep:32B | 1200000개 | 881.52초 | 23396 | 26.5 | 3 |
| Deepseek-R1:32B | 700000개 | 317.70초 | 8487 | 26.7 | 2 |

## 2번 문제 (Caption Paraphrasing)

| **Model** | **정답 여부(60/60/80/80)** | **처리 시간** | **추론 길이** | **단위 시간당 추론(char/sec)** | **Rank** |
| --- | --- | --- | --- | --- | --- | --- |
| o3-mini-high | 60/60/80/80 (맞춤) | 21초 | 0 | N.A | 1 |
| Grok3 | 60/60/80/80 (맞춤) | 93초 | 36220 | 389.5 | 3 |
| QWQ:32B | 60/60/80/80 (맞춤) | 180.73초 | 6498 | 36.0 | 2 |
| EXAONE-Deep:32B | 33/33/53/53 (아예 틀림) | 211.85초 | 9243 | 43.6 | 5 |
| Deepseek-R1:32B | 60/60/80/79 (거의 맞춤) | 152.59초 | 5874 | 38.5 | 4 |

## 3번 문제 (수능 수학 문제)

| **Model** | **정답 여부(21)** | **처리 시간** | **추론 길이** | **단위 시간당 추론(char/sec)** | **Rank** |
| --- | --- | --- | --- | --- | --- | --- |
| o3-mini-high | 23 (절반 맞춤) | 55 | 1413 | 25.7 | 1 |
| Grok3 | 23 (절반 맞춤) | 97 | 30894 | 318.5 | 4 |
| QWQ:32B | 23 (절반 맞춤) | 1081.25 | 29156 | 27.0 | 2 |
| EXAONE-Deep:32B | 23 (절반 맞춤) | 1265.00 | 30510 | 24.1 | 3 |
| Deepseek-R1:32B | 50 (아예 틀림) | 2109.8 | 46070 | 21.8 | 5 |

## 4번 문제 (코딩 문제)

| **Model** | **정답 여부 (54)** | **처리 시간** | **추론 길이** | **단위 시간당 추론(char/sec)** | **Rank** |
| --- | --- | --- | --- | --- | --- | --- |
| o3-mini-high |  54/54 | 57 | 1625 | 28.5 | 1 |
| Grok3 | 36/54 | 380 | 118320 | 311.4 | 3 |
| QWQ:32B | 28/54 | 2053.40 | 65257 | 31.8 | 4 |
| EXAONE-Deep:32B | 51/54 | 1952.09 | 57913 | 29.7 | 2 |
| Deepseek-R1:32B | 24/54 | 1152.13 | 41756 | 36.2 | 5 |

오픈 소스가 아닌 모델은 서버에서 처리되는데, 이때 GPU 사양에 따라 처리 속도가 다를 수 있기 때문에 단위 시간당 추론(char/sec)를 통해 어느 정도 동일 선상에 두고 비교하고자 하였다. 추론 속도는 Grok3가 나머지의 거의 10배 정도의 속도를 내는 것을 볼 수 있고, 추론 길이는 압도적으로 o3-mini가 효과적인 것을 볼 수 있다.

## 결론

각각 task에서의 랭킹은 다음을 각각 우선순위로 하여 매겼다.

1. 정답을 맞췄는가(혹은 얼마나 근접하였는가)
2. 추론 길이가 얼마나 짧았는가

상대적으로 오픈 소스와 o3-mini-high, Grok3와 같이 서버를 매개로 하는 서비스의 처리 시간과 추론 길이에 대해 엄밀하게 비교하기는 힘들 것 같아서 그랬다.

| **Model** | **사용 후기** |
| --- | --- |
| o3-mini-high | 가장 빠르고 정확함 |
| Grok3 | 중간 속도이고, 적당히 정확함 |
| QWQ:32B | 로컬에서 느리긴 하지만 꽤나 정확함 |
| EXAONE-Deep:32B | 로컬에서 느리긴 하지만 꽤나 정확함 |
| Deepseek-R1:32B | 로컬에서 느리고 그다지 정확하지 않음 |

솔직히 실제로 사용해본 결과 체감상 아직 o3-mini-high가 전반적으로 모든 task를 잘 해낸다는 느낌이 있었다. 그리고 의외로 EXAONE과 QWQ는 실제 사용감이 비슷했고, Grok3는 많이 써보지는 않았지만 어느 정도 평타는 치는 듯 했다.  
딥시크의 경우 원본 모델 대신 32B 모델을 사용하여 실험해보았는데, 코딩이나 수학 문제에 대해 생각보다 너무 낮은 정확도를 보이는 것을 확인하였다.

## 공정한 비교가 맞는가?

당연히 공정한 비교는 아니다. 개인적으로 선정한 문제들도 그렇고, 아무래도 추론 모델에 대한 명확한 지표가 아직 나오지 않은 것이 큰 이유라 생각한다.  
이런저런 보다 많은 문제들에 적용해서 직접 사용해봐야 조금 더 공정한 비교가 가능하겠다.  
겨우 4문제 가지고 모델별 순위를 결정하는것 자체가 의미가 없는 일이지만 실제 오픈 소스를 GPU에 올려서 사용해보고 느낀 점은 openai의 서비스가 아직까지는 다른 모델에 비해 좋다는 것이다.  
그리고 오픈 소스 모델을 개인적으로 테스트해보기에는 문제에 따라서 추론 시간이 30분에 육박하는 경우도 있어 쉽지가 않았다.  
서버용 GPU를 달고 최적화를 해야 서비스에 활용할 수 있을 것 같다.

## 추론 모델이 짱짱맨?

과연 추론 모델을 사용하는 것이 문제에 대해 잘 대응할 수 있는 최적의 방식인가? gpt-4.5를 사용해서 동일하게 4개의 문제를 풀려보았고, 각각 결과는 다음과 같다.

| **Model** | Q1 (골프공 갯수) | Q2 (Paraphrasing) | Q3 (수학 문제) | Q4 (코딩 문제) |
| --- | --- | --- | --- | --- |
| gpt-4.5 | 1.4million | 60/60/80/80 | 9 | 54/54 |

추론 모델이 아닌 **gpt-4.5**는 골프공 갯수, paraphrasing과 같은 문제에 대해 **추론 모델과 다를바 없는 성능을 보여주었고**, 수학 문제를 완전히 틀리기는 했지만 **코딩 문제의 모든 테스트 케이스를 통과하는 결과를 내놓았다**. 심지어 추론 모델에 비해 생성되는 속도도 훨씬 빠른 것이 장점이다.

결국 앞으로 추론 모델을 통해 찾아야할 문제는 ‘추론 모델’이 기존의 기본 LLM을 뛰어넘는다는 것이 아니고, 추론 모델만의 특성을 활용할 수 있는 task가 무엇인지 진지하게 생각해봐야할 것이다.

30분 기다렸는데 내뱉은 코드에 오히려 오류가 발생하고, 30초 기다렸는데 내뱉은 코드에는 오류가 없는 현상이 내가 실험한 문제들만 아니라 실생활에 적용될 다양한 문제들에도 동일하게 문제가 될 부분이다. 물론 단순한 알고리즘 문제가 아니라 구현의 단계에서는 추론 모델이 훨씬 효과적으로 작동한다(실제로 본인 경험상).

## 오픈 소스 모델의 발전 및 개인적인 의견

LG가 굉장히 멋진 일을 해냈다고 생각이 든다.  
국내에서 추론 모델이 얼마나 빨리 출시될 수 있을지 궁금했는데, 이렇게 빠르게 성능 높은 모델을 출시해서 뉴스를 보고 놀랐다.  
앞으로도 오픈 소스는 본인을 포함한 수많은 엔지니어들의 피드백을 받을 것이고, 더욱 발전해 나갈 것이라 믿어 의심치 않는다.  
그러나 너무 아쉬웠던 점은 바로 라이센스다. 지표 상으로 높은 점수를 찍긴 했지만 사실 LG에서 엑사원 자체적으로 웹 서비스가 나와도 적당한 task에 사용하라고 하면 굳이 쓰진 않을 것 같다는 생각을 했다. 여전히 chatgpt가 더 좋은 답변을 빠르고 정확하게 내주고 있기 때문이다.  
그렇다면 상대적으로 오픈 소스의 장점인 다양한 실험, 튜닝 및 수많은 서비스 적용을 통한 추가 가치 창출이 경쟁력이 될 것 같은데, 딥시크나 QWQ와 같은 모델이 MIT/Apache 라이센스를 주는 것에 비해 엑사원은 그러지 못했다.  