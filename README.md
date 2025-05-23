# Retrieval-система

# Задача 1

* Реализовал метрики Recall@K, MMR

# Задача 2

* В качестве бейзлайна обучил и протестировал TF-IDF
* Получил метрики:

|Metric   | Value|
|---------|------|
|Recall@1 |0.4104|
|Recall@3 |0.6138|
|Recall@10|0.7849|
|MRR      |0.5385|

# Задача 3

* Имлементировал Retrieval-система, основанную на E5
* Протестировал её на тестовых данных, получил метрики:

|Metric   | Value|
|---------|------|
|Recall@1 |0.5978|
|Recall@3 |0.7999|
|Recall@10|0.9069|
|MRR      |0.7117|

Данный подход сильно лучше TF-IDF бейзлайна, т.к. вектора, полученные из E5 — контекстуализрованные, следовательно в каждый вектор заложен определенный смысл и контекст, в следствие, чего похожие слова имеют похожие вектора, а непохожие, наоборот.


# Задача 4

* Дообучил E5 на triplet и contrastive лоссах, используя рандомизированный майнинг негативов для triplet лосса.
* После обучения, получил метрики на тестовых данных:


# Triplet Loss
|Metric   |Initial| 50 steps | 100 steps | 150 steps| 
|---------|------ |----------|-----------|----------|
|Recall@1 |0.5978 |  0.5958  |  0.5898   |  0.5777  |
|Recall@3 |0.7999 |  0.7981  |  0.7903   |  0.7764  |
|Recall@10|0.9069 |  0.9044  |  0.8986   |  0.8867  |
|MRR      |0.7117 |  0.7102  |  0.7035   |  0.6916  |


Обучение триплет лосса оказалось неудачным, т.к. уже на ранних степах обучения, метрики качества начинали падать, а при обучении на одной эпохе и более метрики становились близки к нулю. Предполагаю, это связано с тем, что выбранная стратегия майнинга негативов плохо работает в подобных задачах, т.к. рандомный выбор негативов может выбирать заведомо далекие объекты от позитива объекты, в последствие чего модель не учится, а деградирует.

# Contrastive Loss

|Metric   |Initial| 150 steps | 300 steps | 450 steps| 600 steps|
|---------|------ |----------|-----------|----------|-----------|
|Recall@1 |0.5978 |  0.6078  |  0.6181   |  0.6143  |   0.5943  |
|Recall@3 |0.7999 |  0.8096  |  0.8198   |  0.8176  |   0.7939  |
|Recall@10|0.9069 |  0.9129  |  0.9224   |  0.9221  |   0.9052  |
|MRR      |0.7117 |  0.7211  |  0.7312   |  0.7287  |   0.7083  |

Обучение с контрастив лоссом имеет определенные успехи, и улучшает базовые метрики в среднем на 0.02, при этом ближе к середине обучения начинает переобучаться и качество модели падает.



В целом модель обученная на Contrastive Loss улучшает качество базовой модели, т.к. данный лосс более стабилен + не требует майнинга негативов, Triplet Loss не показал положительных результатов, в связи с тем, что выбранная стратегия подбора негативов очень слаба и не даёт достаточного импульса модели для успешного обучения.


# Задача 5
# Triplet Loss with Hard Negatives

