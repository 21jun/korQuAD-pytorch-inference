# -*- coding:utf-8 -*-
from bert import QA

model = QA('model')

# doc = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the state's law-making body for matters coming under state responsibility. The Victorian Constitution can be amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian people in a referendum, depending on the provision."
doc = """
배터리 용량 은 내장 형 3600 mAh 이 다 . 
이 는 디스플레이 크기 를 키우 면서 내부 적 으로 활용 할 수 있 는 공간 이 늘 었 고 기기 두께 도 상대 적 으로 두꺼워 졌 기 때문 으로 전작 인 갤럭시 S 6 엣지 와 비교 할 때 약 1000 mAh 가량 증가 한 수 치이 며 갤럭시 S 6 엣지 + 와 비교 할 때 도 약 600 mAh 가량 증가 한 수치 로 , 
배터리 타임 은 갤럭시 노트 4 대비 약 70 % 정도 오른 수준 으로 체감 된다고 한다 . 
또한 , 전작 인 갤럭시 S 6 엣지 와 마찬가지 로 삼성전자 Adaptive fast charging 고속 충전 솔루션 과 자기 유도 방식 인 Qi 규격 과 자기 유도 방식 이나 , 
최근 자기 공진 방식 의 A 4 WP 와 호환 성 을 강화 한 PMA 규격 을 만족 하 는 무선 충전 솔루션 을 내장 했으며 갤럭시 S 6 엣지 + 의 고속 무선 충전 기술 까지 도입 했 다 . 
하지만 누가 업데이트 이후 로 노트 7 사건 으로 출력 량 을 완전히 줄여 버려서 고속 무선 충전 이 라는 말 도 의미 없 어 진 것 같 다 .
"""


q = "배터리 타임은 갤럭시 노트4 대비 얼마나 체감 되는가?"

answer = model.predict(doc, q)
print("Question:", q)
print("Answer  :", answer['answer'])
