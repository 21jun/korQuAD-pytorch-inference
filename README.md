# BERT-KorQuad inference API

> references

forked from :   https://github.com/kamalkraj/BERT-SQuAD 
data from   :   https://korquad.github.io/category/1.0_KOR.html



# Result

`model` : bert-base-multilingual-cased

`epochs`: 1 time

```json
{'HasAns_exact': 68.99896085902321,
 'HasAns_f1': 76.5909835844023,
 'HasAns_total': 5774,
 'best_exact': 68.99896085902321,
 'best_exact_thresh': 0.0,
 'best_f1': 76.5909835844023,
 'best_f1_thresh': 0.0,
 'exact': 68.99896085902321,
 'f1': 76.5909835844023,
 'total': 5774}
```

# Train your model with KorQuad Dataset

## via Colab

https://drive.google.com/file/d/1bqINXkKZcp7eZEGHuegmilJ7KKnWhWqT/view?usp=sharing



1. Download KorQuad dataset from [here](https://korquad.github.io/category/1.0_KOR.html)

    - `KorQuAD_v1.0_train.json`
    - `KorQuAD_v1.0_dev.json`

2. Upload dataset to Google Drive and copy the path

    ex) `'My Drive'/develop/pytorch-KorQuAD/korquad/KorQuAD_v1.0_train.json`

3. Follow the Colab notebook file.


4. copy `outputs` files to `model` directory in this project


## via Local Machine

1. Download KorQuad dataset from [here](https://korquad.github.io/category/1.0_KOR.html)

    - `KorQuAD_v1.0_train.json`
    - `KorQuAD_v1.0_dev.json`

2. Run `training\run_squad.py` with arguments below

    ex)
    ```sh
    python3 training\run_squad.py
        --model_type bert 
        --data_dir ./
        --model_name_or_path bert-base-multilingual-cased
        --do_train
        --do_eval
        --output_dir ./outputs
        --predict_file ./KorQuAD_v1.0_dev.json
        --train_file ./KorQuAD_v1.0_train.json
        --save_steps 10000
        --num_train_epochs 
    ```

3. copy `outputs` files to `model` directory in this project


# Run in Script

```sh
python3 main.py
```

```py
# -*- coding:utf-8 -*-
from bert import QA
from konlpy.tag import Okt

model = QA('model')

doc = """
유럽의 정치 경제 통합을 실현하기 위하여 1993년 11월 1일 발효된 마스트리히트 조약에 따라 유럽 12개국이 참가하여 출범한 연합 기구다. 2020년 2월 기준, 가입국은 27개국이다. 27개국을 모두 합치면 인구는 약 5억, 경제 규모는 미국과 맞먹는 거대한 집단이라서 세계 주요 정치, 외교, 안보, 경제, 사회, 환경 현안에서도 EU 집행위원장은 강대국의 대통령과 버금가는 대우를 받는다. 현재 지구상에서 가장 구속력 있게 단결되어 있는 국가집단이라고 할 수 있다. 유로화는 특별인출권에서 미국 달러 다음으로 2위의 비율을 차지하고 있다.현재 EU가 세계에서 총 GDP 측면에서 가장 큰 경제권은 아닌데, 브렉시트 이전인 2015년 시점에서 유럽연합 GDP는 이미 미국 GDP에 추월당했다. 창립 이후부터 2010년대 초반까지는 전체 경제규모에서 미국을 능가하며 절대 무시 못할 영향력을 지녔었지만, 2008년 리먼 브라더스 사태와 이후 유로존 사태로 이어지는 만성적인 경제 불황으로 인해 미국에 추월당했다. 2020년 1월 31일 영국의 탈퇴로 EU의 경제 규모는 더 줄어들었다.모든 유럽연합 가입 국가들은 민주주의를 도입하고 사형을 폐지해야 한다. 민주주의, 인권 강조는 마스트리흐트 조약 제2관 공동 외교안보 정책 분야에 세계 인권 선언을 각국의 헌법 원리로 수용하고 명시적인 비준 동의를 할 것을 전제로 성립되어 있다.
"""

q = "유럽연합의 GDP를 추월한 국가는?"

okt = Okt()
# doc = ' '.join(okt.morphs(doc))  # 정답에 조사를 제거하고 싶다면...
# q = ' '.join(okt.morphs(q))

answer = model.predict(doc, q)
print("Question:", q)
print("Answer  :", answer['answer'])
```



# Serving with Flask

```sh
python3 api.py
```
API will be live at `0.0.0.0:8000` endpoint `predict`

### cURL
```sh
curl --location --request POST 'http://localhost:8000/predict' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer 5a965cd7-0ec3-4312-a7aa-dc8da4838e18' \
--data-raw '{
	"document":"유럽의 정치 경제 통합을 실현하기 위하여 1993년 11월 1일 발효된 마스트리히트 조약에 따라 유럽 12개국이 참가하여 출범한 연합 기구다. 2020년 2월 기준, 가입국은 27개국이다. 27개국을 모두 합치면 인구는 약 5억, 경제 규모는 미국과 맞먹는 거대한 집단이라서 세계 주요 정치, 외교, 안보, 경제, 사회, 환경 현안에서도 EU 집행위원장은 강대국의 대통령과 버금가는 대우를 받는다. 현재 지구상에서 가장 구속력 있게 단결되어 있는 국가집단이라고 할 수 있다. 유로화는 특별인출권에서 미국 달러 다음으로 2위의 비율을 차지하고 있다.현재 EU가 세계에서 총 GDP 측면에서 가장 큰 경제권은 아닌데, 브렉시트 이전인 2015년 시점에서 유럽연합 GDP는 이미 미국 GDP에 추월당했다. 창립 이후부터 2010년대 초반까지는 전체 경제규모에서 미국을 능가하며 절대 무시 못할 영향력을 지녔었지만, 2008년 리먼 브라더스 사태와 이후 유로존 사태로 이어지는 만성적인 경제 불황으로 인해 미국에 추월당했다. 2020년 1월 31일 영국의 탈퇴로 EU의 경제 규모는 더 줄어들었다.모든 유럽연합 가입 국가들은 민주주의를 도입하고 사형을 폐지해야 한다. 민주주의, 인권 강조는 마스트리흐트 조약 제2관 공동 외교안보 정책 분야에 세계 인권 선언을 각국의 헌법 원리로 수용하고 명시적인 비준 동의를 할 것을 전제로 성립되어 있다.",
	"question":"마스트리히트 조약이 발효된 시기는?"
}'
```

```json
{
    "result": {
        "answer": "1993년 11월 1일",
        "confidence": 0.988371894980254,
        "document": [
            "유럽의",
            "정치",
            "경제",
            "통합을",
            "실현하기",
            "위하여",
            "1993년",
            "11월",
            "1일",
            "발효된",
            "마스트리히트",
            "조약에",
            "따라",
            "유럽",
            "12개국이",
            "참가하여",
            "출범한",
            "연합",
            "기구다.",
            "2020년",
            "2월",
            "기준,",
            "가입국은",
            "27개국이다.",
            "27개국을",
            "모두",
            "합치면",
            "인구는",
            "약",
            "5억,",
            "경제",
            "규모는",
            "미국과",
            "맞먹는",
            "거대한",
            "집단이라서",
            "세계",
            "주요",
            "정치,",
            "외교,",
            "안보,",
            "경제,",
            "사회,",
            "환경",
            "현안에서도",
            "EU",
            "집행위원장은",
            "강대국의",
            "대통령과",
            "버금가는",
            "대우를",
            "받는다.",
            "현재",
            "지구상에서",
            "가장",
            "구속력",
            "있게",
            "단결되어",
            "있는",
            "국가집단이라고",
            "할",
            "수",
            "있다.",
            "유로화는",
            "특별인출권에서",
            "미국",
            "달러",
            "다음으로",
            "2위의",
            "비율을",
            "차지하고",
            "있다.현재",
            "EU가",
            "세계에서",
            "총",
            "GDP",
            "측면에서",
            "가장",
            "큰",
            "경제권은",
            "아닌데,",
            "브렉시트",
            "이전인",
            "2015년",
            "시점에서",
            "유럽연합",
            "GDP는",
            "이미",
            "미국",
            "GDP에",
            "추월당했다.",
            "창립",
            "이후부터",
            "2010년대",
            "초반까지는",
            "전체",
            "경제규모에서",
            "미국을",
            "능가하며",
            "절대",
            "무시",
            "못할",
            "영향력을",
            "지녔었지만,",
            "2008년",
            "리먼",
            "브라더스",
            "사태와",
            "이후",
            "유로존",
            "사태로",
            "이어지는",
            "만성적인",
            "경제",
            "불황으로",
            "인해",
            "미국에",
            "추월당했다.",
            "2020년",
            "1월",
            "31일",
            "영국의",
            "탈퇴로",
            "EU의",
            "경제",
            "규모는",
            "더",
            "줄어들었다.모든",
            "유럽연합",
            "가입",
            "국가들은",
            "민주주의를",
            "도입하고",
            "사형을",
            "폐지해야",
            "한다.",
            "민주주의,",
            "인권",
            "강조는",
            "마스트리흐트",
            "조약",
            "제2관",
            "공동",
            "외교안보",
            "정책",
            "분야에",
            "세계",
            "인권",
            "선언을",
            "각국의",
            "헌법",
            "원리로",
            "수용하고",
            "명시적인",
            "비준",
            "동의를",
            "할",
            "것을",
            "전제로",
            "성립되어",
            "있다."
        ],
        "end": 8,
        "start": 6
    }
}
```

## Postman

![postman](img/postman.png)