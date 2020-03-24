# -*- coding:utf-8 -*-
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForQuestionAnswering, BertTokenizer)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

model_path = "model"


tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
# tokenizer = BertTokenizer("model/vocab copy.txt", do_lower_case=False)
# tokenizer = BertTokenizer("test_vocab.txt", do_lower_case=False)
# a = tokenizer.tokenize("유럽 연합 과 유럽 시민 권 은 1993 년 마스트리흐트 조약 이 발효 되 었 을 때 설립 되 었 다 . 유럽 연합 의 기원 은 1951 년 파리 조약 으로 창설 된 유럽 석탄 철강 공동체 와 1957 년 로마 조약 으로 창설 된 유럽 경제 공동체 이 다 . 유럽 의 공동체 를 창립 한 것 으로 알려진 회원국 은 벨기에 , 프랑스 , 이탈리아 , 네덜란드 , 룩셈부르크 , 서독 6 개국 으로 , 이 들 을 합쳐 이너 식스 라고 부른다 . 위원회 와 그 들 의 계승자 들 은 새로운 회원국 이 가입 하 면서 확장 되 었 다 . 유럽 연합 의 헌법 적 기초 에 대한 마지막 주요 합의 사항 은 2009 년 리스본 조약 에서 체결 되 었 다 . 현재 까지 유럽 연합 에서 탈퇴 한 국가 는 영국 으로 , 2020 년 탈퇴 하 였 다 .")
b = tokenizer.tokenize("2016 년 에 갤럭시 S7 엣지 가 폭발한 사건 은 어느 지역 에서 일어 났는 가?")
c = tokenizer.convert_tokens_to_ids(b)
print(b)
print(c)
