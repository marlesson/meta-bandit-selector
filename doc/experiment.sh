python -m unittest tests/

python pack.py --config-path config_egreedy.yml --polity-module policy.e_greedy --polity-cls EGreedyPolicy

python pack.py --config-path config_softmax.yml --polity-module policy.softmax --polity-cls SotfmaxPolicy

bentoml serve MetaBanditClassifier:latest
bentoml serve RandomRankingRecommender:latest --port 5001
bentoml serve MostPopularRankingRecommender:latest --port 5002
bentoml serve CVAERankingRecommender:latest --port 5003

# bentoml serve MostPopularPerUserRankingRecommender:latest --port 5002
# bentoml serve MatrixFactorizationRankingRecommender:latest --port 5003

# bentoml serve CDAERankingRecommender:latest --port 5005

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy meta_bandit --bandit-policy-params '{"endpoints": "http://localhost:5000"}' --obs-batch-size 1 --val-size 0 --obs "MetaBandit 20"