import pandas as pd

from search_backend_via_bucket import *
from itertools import islice
import json
from time import time


def map_at_40(retrieved_documents, relevant_documents):
    retrieved_documents = retrieved_documents[:40]
    relevant_documents = set(relevant_documents)
    precision_sum = 0.0
    num_relevant = 0
    for i, doc in enumerate(retrieved_documents):
        if doc in relevant_documents:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    if num_relevant == 0:
        return 0.0
    return precision_sum / min(num_relevant, 40)


def recall_at_40(retrieved_documents, relevant_documents):
    retrieved_documents = retrieved_documents[:40]
    relevant_documents = set(relevant_documents)
    retrieved_relevant = [d for d in retrieved_documents if d in relevant_documents]
    try:
        return len(retrieved_relevant) / min(len(relevant_documents),40)
    except ZeroDivisionError:
        return 0


def recall(retrieved_documents, relevant_documents):
    relevant_documents = set(relevant_documents)
    retrieved_relevant = [d for d in retrieved_documents if d in relevant_documents]
    try:
        return len(retrieved_relevant) / len(relevant_documents)
    except ZeroDivisionError:
        return 0


def precision(retrieved_documents, relevant_documents):
    relevant_documents = set(relevant_documents)
    retrieved_relevant = [d for d in retrieved_documents if d in relevant_documents]
    try:
        return len(retrieved_relevant) / len(retrieved_documents)
    except ZeroDivisionError:
        return 0


def f_measure(retrieved_documents, relevant_documents):
    p = precision(retrieved_documents, relevant_documents)
    r = precision(retrieved_documents, relevant_documents)
    try:
        return (2*p*r) / (p+r)
    except ZeroDivisionError:
        return 0


def mmr(retrieved_documents, relevant_documents):
    for i, doc in enumerate(retrieved_documents):
        if doc in relevant_documents:
            return 1/(i+1)
    return 0


def Model1(q):
    tokens = [token.group() for token in RE_WORD.finditer(q.lower()) if token.group() not in all_stopwords and token.group() in index_title.df.keys()]
    tokens = list(np.unique(tokens))
    terms_wights = {term: 1 for term in tokens}
    res = sorted([(k,v) for k,v in candidates_by_title_or_anchore(tokens,terms_wights, index_title).items()],key=lambda x:x[1],reverse=True)
    res = [(int(i[0]), DT.get(i[0], "No-Title")) for i in res][:10]
    return res


def Model2(q, w = 0.5):
    org_tokens = [token.group() for token in RE_WORD.finditer(q.lower())]
    tokens = [token for token in org_tokens if token not in all_stopwords and token in index_body.df.keys()]
    terms_wights = {term: 1 for term in tokens}
    score_title = candidates_by_title_or_anchore(tokens,terms_wights, index_title,binary=False)
    res = sorted([(k, w*v+(1-w)*NPR.get(k, 0)) for k, v in score_title.items()],key=lambda x: x[1],reverse=True)
    res = [(i[0], DT[i[0]]) for i in res][:10]
    return res


def Model3(q, w1 = 0.75, w2 = 1, w3 = 0.25, w4 = 0.1):
    org_tokens = [token.group() for token in RE_WORD.finditer(q.lower())]
    tokens = [token for token in org_tokens if token not in all_stopwords and token in index_body.df.keys()]
    terms_wights = {term: 1 for term in tokens}
    score_title = candidates_by_title_or_anchore(tokens,terms_wights, index_title)
    score_body = candidates_by_body(tokens, terms_wights, index_body)
    score_anchor = candidates_by_title_or_anchore(tokens,terms_wights, index_anchor)
    res = []
    docs = np.unique(list(score_title.keys()) + list(score_body.keys()))
    for d in docs:
        res.append((d,w1*score_title.get(d,0)+w2*score_body.get(d,0)+w3*score_anchor.get(d,0)+w4*NPR.get(d,0)))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = [(i[0], DT[i[0]]) for i in res][:10]
    return res


def Model4(q, w1 = 0.75, w2 = 1, w3 = 0.25, w4 = 0.1):
    org_tokens = [token.group() for token in RE_WORD.finditer(q.lower())]
    tokens = [token for token in org_tokens if token not in all_stopwords and token in index_body.df.keys()]
    terms_wights = {term: 1 for term in tokens}

    if len(tokens) == 1:
        for term in find_top_n_similar_words(tokens[0],1):
            if term in index_body.df.keys():
                tokens.append(term)
                terms_wights[term] = 0.05
    score_title = candidates_by_title_or_anchore(tokens, terms_wights, index_title)
    score_body = candidates_by_body(tokens, terms_wights, index_body)
    score_anchor = candidates_by_title_or_anchore(tokens, terms_wights, index_anchor)
    res = []
    docs = np.unique(list(score_title.keys()) + list(score_body.keys()))
    for d in docs:
        res.append((d,w1*score_title.get(d,0)+w2*score_body.get(d,0)+w3*score_anchor.get(d,0)+w4*NPR.get(d,0)))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = [(i[0], DT[i[0]]) for i in res][:10]
    return res


def Model5(q, w1 = 0.75, w2 = 1, w3 = 0.25, w4 = 0.1):
    org_tokens = [token.group() for token in RE_WORD.finditer(q.lower())]
    tokens = [token for token in org_tokens if token not in all_stopwords and token in index_body.df.keys()]
    stem_tokens = [stemmer.stem(token) for token in tokens]
    terms_wights = {term: 1 for term in tokens}
    terms_wights_with_stem = {term: 1 for term in stem_tokens}
    if len(tokens) == 1:
        for term in find_top_n_similar_words(tokens[0],1):
            if term in index_body.df.keys():
                tokens.append(term)
                stem_tokens.append(stemmer.stem(term))
                terms_wights_with_stem[stemmer.stem(term)] = 0.05
                terms_wights[term] = 0.05

    score_body = candidates_by_body(tokens,terms_wights,index_body)
    score_anchor = candidates_by_title_or_anchore(tokens, terms_wights, index_anchor)
    score_title = candidates_by_title_or_anchore(stem_tokens, terms_wights_with_stem, index_stem_title)
    res = []
    docs = np.unique(list(score_title.keys()) + list(score_body.keys()))
    for d in docs:
        res.append((d,w1*score_title.get(d,0)+w2*score_body.get(d,0)+w3*score_anchor.get(d,0)+w4*NPR.get(d,0)))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = [(i[0], DT[i[0]]) for i in res][:10]
    return res


def Model6(q,w1 = 1, w2 = 1, w3 = 0.25, w4 = 1):
    org_tokens = [token.group() for token in RE_WORD.finditer(q.lower())]

    tokens = [token for token in org_tokens if token not in all_stopwords and token in index_body.df.keys()]

    stem_tokens = [stemmer.stem(token) for token in tokens]
    terms_wights = {term: 1 for term in tokens}
    terms_wights_with_stem = {term: 1 for term in stem_tokens}

    if len(tokens) == 1:
        for term in find_top_n_similar_words(tokens[0],1):
            if term in index_body.df.keys():
                tokens.append(term)
                stem_tokens.append(stemmer.stem(term))
                terms_wights_with_stem[stemmer.stem(term)] = 0.05
                terms_wights[term] = 0.05
    if len([term for term in org_tokens if term in location_queries['all']]) > 0:
        for term in location_queries['city and country']:
            if term not in tokens:
                tokens.append(term)
                terms_wights[term] = 0.05
        score_body = candidates_by_body(tokens,terms_wights, index_body)
        res = []
        for k,v in score_body.items():
            res.append((k,score_body.get(k, 0) + PR.get(k, 0)))
        res = sorted(res, key=lambda x: x[1], reverse=True)
        res = [(i[0], DT[i[0]]) for i in res][:45]
        return res
    score_body = candidates_by_body(tokens,terms_wights, index_body)
    score_anchor = candidates_by_title_or_anchore(tokens, terms_wights, index_anchor)
    score_title = candidates_by_title_or_anchore(stem_tokens, terms_wights_with_stem, index_stem_title)
    res = []
    docs = np.unique(list(score_title.keys()) + list(score_body.keys()))
    for d in docs:
        res.append((d,w1*score_title.get(d,0)+w2*score_body.get(d,0)+w3*score_anchor.get(d,0)+w4*NPR.get(d,0)))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = [(int(i[0]), DT[i[0]]) for i in res][:10]
    return res


def Model7(q, w1 = 1, w2 = 1, w3 = 0.25, w4 = 1,wn = 1.16):
    org_tokens = [token.group() for token in RE_WORD.finditer(q.lower())]
    pos_tagged = nltk.pos_tag(org_tokens)

    # Extracting Nouns
    nouns = np.unique([word.lower() for word, tag in pos_tagged if tag.startswith("N")])
    tokens = [token for token in org_tokens if token not in all_stopwords and token in index_body.df.keys()]

    stem_tokens = [stemmer.stem(token) for token in tokens]
    terms_wights = {term: 1 for term in tokens}
    terms_wights_with_stem = {term: 1 for term in stem_tokens}
    for token in tokens:
        if token in nouns:
            terms_wights_with_stem[stemmer.stem(token)] = wn
            terms_wights[token] = wn
    if len(tokens) == 1:
        for term in find_top_n_similar_words(tokens[0],1):
            if term in index_body.df.keys():
                tokens.append(term)
                stem_tokens.append(stemmer.stem(term))
                terms_wights_with_stem[stemmer.stem(term)] = 0.05
                terms_wights[term] = 0.05
    if len([term for term in org_tokens if term in location_queries['all']]) > 0:
        for term in location_queries['city and country']:
            if term not in tokens:
                tokens.append(term)
                terms_wights[term] = 0.05
        score_body = candidates_by_body(tokens,terms_wights, index_body)
        res = []
        for k,v in score_body.items():
            res.append((k,score_body.get(k, 0) + PR.get(k, 0)))
        res = sorted(res, key=lambda x: x[1], reverse=True)
        res = [(i[0], DT[i[0]]) for i in res][:10]
        return res
    score_body = candidates_by_body(tokens,terms_wights, index_body)
    score_anchor = candidates_by_title_or_anchore(tokens, terms_wights, index_anchor)
    score_title = candidates_by_title_or_anchore(stem_tokens, terms_wights_with_stem, index_stem_title)
    res = []
    docs = np.unique(list(score_title.keys()) + list(score_body.keys()))
    for d in docs:
        res.append((d,w1*score_title.get(d,0)+w2*score_body.get(d,0)+w3*score_anchor.get(d,0)+w4*NPR.get(d,0)))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = [(int(i[0]), DT[i[0]]) for i in res][:10]
    return res


with open('queries_train.json', 'rt') as f:
    queries = json.load(f)

train_set = list(islice(queries.items(), 0,20))
test_set = list(islice(queries.items(), 20,30))

score = {"Model No'": [1, 2, 3, 4, 5, 6, 7], 'MAP@40': [], 'Recall at 40': [],'Duration': [], 'Recall': [], 'Precision': [], 'MRR': [], 'F-Measure': []}
models = [Model1, Model2, Model3, Model4, Model5, Model6, Model7]
for model in models:
    local = {'MAP@40': [], 'Recall at 40': [], 'Duration': [], 'Recall': [], 'Precision': [], 'MRR': [],
             'F-Measure': []}
    for q in queries.items():
        t_start = time()
        real = q[1]
        pred = [i[0] for i in model(q[0])]
        duration = time() - t_start
        local['MAP@40'].append(map_at_40(pred, real))
        local['Recall at 40'].append(recall_at_40(pred, real))
        local['Duration'].append(duration)
        local['Recall'].append(recall(pred, real))
        local['Precision'].append(precision(pred, real))
        local['MRR'].append(mmr(pred, real))
        local['F-Measure'].append(f_measure(pred, real))
    score['MAP@40'].append(np.average(local['MAP@40']))
    score['Recall at 40'].append(np.average(local['Recall at 40']))
    score['Duration'].append(np.average(local['Duration']))
    score['Recall'].append(np.average(local['Recall']))
    score['Precision'].append(np.average(local['Precision']))
    score['MRR'].append(np.average(local['MRR']))
    score['F-Measure'].append(np.average(local['F-Measure']))

df = pd.DataFrame(score)
df.to_csv("Evaluation.csv")


local = {"Query": [], 'MAP@40': [], 'Recall at 40': [], 'Duration': [], 'Recall': [], 'Precision': [], 'MRR': [],
             'F-Measure': []}
for q in queries.items():
    t_start = time()
    real = q[1]
    pred = [i[0] for i in Model7(q[0])]
    duration = time() - t_start
    local['Query'].append(q[0])
    local['MAP@40'].append(map_at_40(pred, real))
    local['Recall at 40'].append(recall_at_40(pred, real))
    local['Duration'].append(duration)
    local['Recall'].append(recall(pred, real))
    local['Precision'].append(precision(pred, real))
    local['MRR'].append(mmr(pred, real))
    local['F-Measure'].append(f_measure(pred, real))

df = pd.DataFrame(local)
df.to_csv("Queries_evaluation.csv")
