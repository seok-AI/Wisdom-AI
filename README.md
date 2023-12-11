# Wisdom-AI

## Content
1. [Performance comparison](#Performance-comparison)
2. [competition](#weight)
4. [Reference](#Reference)

## Performance comparison
|code|test mape|model|task type|hyperparameter|drop columns|Validation|
|---|---|---|---|---|---|---|
|AutoML.ipynb|0.5133|autogluon|Table|best_quality, num_stack: 3|filename, ID, date, weight, bmi|No|
|MultiModal.py|0.4304|swinv2 + mlp|Multimodal|PolynomialFeatures(degree=2)|ID, date, filename|Yes|
|HEQ.py|0.4595|swinv2 + mlp|Multimodal|HEQ|ID, date, filename|Yes|


## competition
[Track 2] 2023 바이오헬스 데이터 경진대회 - 치의학 분야 (일반 부문)


![image](https://github.com/seok-AI/Wisdom-AI/assets/85815265/64a91b4b-b6c6-4590-9770-824d5de103de)


## Reference
https://aiconnect.kr/competition/detail/233
