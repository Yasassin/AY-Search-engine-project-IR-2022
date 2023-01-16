import math

import numpy as np
from flask import Flask, request, jsonify

from inverted_index_gcp import InvertedIndex, MultiFileReader
from inverted_index_gcp_title import InvertedIndex, MultiFileReader
from inverted_index_gcp_anchor import InvertedIndex, MultiFileReader

from contextlib import closing


from collections import Counter, OrderedDict, defaultdict

import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords, wordnet
import pickle
import pandas as pd


import hashlib


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stemmer = PorterStemmer()


with open("postings_gcp/index.pkl", 'rb') as f:
    index_body = pickle.load(f)

with open("postings_gcp_title/index.pkl", 'rb') as f:
    index_title = pickle.load(f)

with open("postings_gcp_anchor/index.pkl", 'rb') as f:
    index_anchor = pickle.load(f)

with open("global/dl.pkl", 'rb') as f:
    DL = pickle.load(f)

with open("global/dt.pkl", 'rb') as f:
    DT = pickle.load(f)

with open("global/pageviews-202108-user.pkl", 'rb') as f:
    pageviews = pickle.load(f)

with open("pr/pr.pkl", 'rb') as f:
    PR = pickle.load(f)

with open("pr/Npr.pkl", 'rb') as f:
    NPR = pickle.load(f)

with open("postings_gcp_stemming_title/index.pkl", 'rb') as f:
    index_stem_title = pickle.load(f)


NUMBER_OF_DOCS = 6348910

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became", "make", "made", "come"]

location_queries ={'all':{'where','nation','location','continent','region','area','navigate','city','town','cities','country','state','countries','place'},
                   'city and country':{'city','country','state'}}

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def generate_query_tfidf_vector(query_to_search,terms_wights, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(index.df)
    Q = np.zeros(total_vocab_size)
    term_vector = list(index.df.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log(NUMBER_OF_DOCS/(df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf * terms_wights[token]
            except:
                pass
    return Q


def get_candidate_documents_and_scores(query_to_search, index):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    doc_per_term = int(300/len(query_to_search))
    for term in np.unique(query_to_search):
        if term in index.df.keys():
            list_of_doc = sorted(index.read(term), key=lambda x: x[1], reverse=True)[:doc_per_term]
            normlized_tfidf = [(doc_id, (freq / DL[doc_id]) * math.log(NUMBER_OF_DOCS/ index.df[term], 10)) for
                               doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search,terms_wights, index):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.


    words,pls: iterator for working with posting.

    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(index.df)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index)
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = index.df.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf * terms_wights[term]

    return D


def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    temp = np.dot(D, Q)
    sqr_df = np.sqrt((D * D).sum(axis=1) * (Q * Q).sum())
    final = temp / sqr_df
    res = dict(final)
    return res


def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]


def candidates_by_body(tokens,terms_wights,index):
    """
    The function receive tokens of query, the wight of each term and the relevant index to use.
    The function returns dict of scores according using cossin similarly with tfidf.
    param tokens: list of strings, the terms from the query.
    param terms_wights: dict string: float, the weight of each token.
    param index: Inverted index, the relevant index to use
    return: dict of string : float, doc_id: score
    """
    tf_idf_q = generate_query_tfidf_vector(tokens,terms_wights, index)
    document_tfidf_matrix = generate_document_tfidf_matrix(tokens,terms_wights, index)
    cosine_similarity_dict = cosine_similarity(document_tfidf_matrix, tf_idf_q)
    return cosine_similarity_dict


def get_top_100_docs_by_body(q):
    """
    The function receive string and retrieve the 100 top docs by tfidf and cossin similarly based on the body of the docs.
    The function give equal weight to each token after the initial filtering of stop words.
    param q: string, the query to search
    return: list of (int,string), list of (doc_id,doc_title) sorted in descending order according to the score.
    """
    tokens = [token.group() for token in RE_WORD.finditer(q.lower()) if token.group() not in all_stopwords]
    terms_wights = {term: 1 for term in tokens}
    res = [(i[0], DT[i[0]]) for i in get_top_n(candidates_by_body(tokens,terms_wights,index_body), 100)]
    return res


def candidates_by_title_or_anchore(query_to_search, terms_wights, index, binary=True):
    """
    The function receive tokens of query, the wight of each term and the relevant index to use.
    The function returns dict of scores according to whether the word apper or not.
    param tokens: list of strings, the terms from the query.
    param terms_wights: dict string: float, the weight of each token.
    param index: Inverted index, the relevant index to use
    param binary: bool, whether to use it as binary score or not.
    return: dict of string : float, doc_id: score
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in index.df.keys():
            list_of_doc = index.read(term)
            if binary:
                for doc_id, freq in list_of_doc:
                    candidates[doc_id] = candidates.get(doc_id, 0) + 1*terms_wights[term]
            else:
                for doc_id, freq in list_of_doc:
                    candidates[doc_id] = candidates.get(doc_id, 0) + freq*terms_wights[term]

    return candidates


def get_top_docs_by_title(q):
    """
    The function receive string and retrieve docs by whether the word apper or not based on the title of the docs.
    The function give equal weight to each token after the initial filtering of stop words.
    param q: string, the query to search
    return: list of (int,string), list of (doc_id,doc_title) sorted in descending order according to the score.
    """
    tokens = [token.group() for token in RE_WORD.finditer(q.lower()) if token.group() not in all_stopwords]
    tokens = list(set(tokens))
    terms_wights = {term: 1 for term in tokens}
    res = sorted([(k,v) for k,v in candidates_by_title_or_anchore(tokens,terms_wights, index_title).items()],key=lambda x:x[1],reverse=True)
    res = [(i[0], DT[i[0]]) for i in res]
    return res


def get_top_docs_by_anchor(q):
    """
    The function receive string and retrieve docs by whether the word apper or not based on the anchor of the docs.
    The function give equal weight to each token after the initial filtering of stop words.
    param q: string, the query to search
    return: list of (int,string), list of (doc_id,doc_title) sorted in descending order according to the score.
    """
    tokens = [token.group() for token in RE_WORD.finditer(q.lower()) if token.group() not in all_stopwords]
    tokens = list(set(tokens))
    terms_wights = {term: 1 for term in tokens}
    res = sorted([(k, v) for k, v in candidates_by_title_or_anchore(tokens,terms_wights, index_anchor).items()], key=lambda x: x[1], reverse=True)
    res = [(i[0], DT.get(i[0], "No-Title")) for i in res]
    return res


def get_page_rank(docs):
    """
    The function receive list of docs and return their matching PageRank score
    Args:
        docs: list of int, list of doc_id

    Returns: list of int, list of PageRank

    """
    return [PR.get(i, 0) for i in docs]


def get_page_views(docs):
    """
    The function receive list of docs and return their matching number of page views
    Args:
        docs: list of int, list of doc_id

    Returns: list of int, list of page views

    """
    return [pageviews.get(i, 0) for i in docs]


def find_top_n_similar_words(word, n):
    """
    The function receive word and return the top n words that are similar to the word by WordNet score
    Args:
        word: string, the word
        n: int, number of words to return

    Returns: list of strings, list of similar words sorted form the most similar to lest similar

    """
    res = []
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                for i in wordnet.synsets(lemma.name()):
                    for j in i.lemmas():
                        similarity = wordnet.wup_similarity(i, wordnet.synset(f'{word}.n.01'))
                        if j.name() != word:
                            res.append((j.name(),similarity))
        res = [i[0] for i in sorted(res,key=lambda x:x[1],reverse=True)][:n]
    except:
        return []
    return res


def AY_Search(q, w1 = 1, w2 = 1, w3 = 0.25, w4 = 1,wn = 1.16):
    """
    The function find the most relevant documents, from Wikipedia corpus, to a specific query based on the query, the
    title, body and anchor of a doc.
    Args:
        q: string, the query to search
        w1: float, The weight of the body scores
        w2: float, The weight of the title scores
        w3: float, The weight of the anchor scores
        w4: float, The weight of the PageRank scores
        wn: float, The weight of nouns

    Returns: list of tuple (int, string), list of relevant documents in the form (doc_id,doc_title) sorted in their relevance
    """
    # Create alist of tokens using regular expression
    org_tokens = [token.group() for token in RE_WORD.finditer(q.lower())]

    # Using nltk to analysis the role of each term in the query
    pos_tagged = nltk.pos_tag(org_tokens)

    # Extracting Nouns
    nouns = np.unique([word.lower() for word, tag in pos_tagged if tag.startswith("N")])

    # Extracting tokens that are not stop words
    tokens = [token for token in org_tokens if token not in all_stopwords and token in index_body.df.keys()]

    # Create list of stemmed tokens
    stem_tokens = [stemmer.stem(token) for token in tokens]

    # initial the weight of the terms to be 1
    terms_wights = {term: 1 for term in tokens}
    terms_wights_with_stem = {term: 1 for term in stem_tokens}

    # increase the weight of nouns
    for token in tokens:
        if token in nouns:
            terms_wights_with_stem[stemmer.stem(token)] = wn
            terms_wights[token] = wn

    # Expan the query of single term with another word with lower weight
    if len(tokens) == 1:
        for term in find_top_n_similar_words(tokens[0],1):
            tokens.append(term)
            stem_tokens.append(stemmer.stem(term))
            terms_wights_with_stem[stemmer.stem(term)] = 0.05
            terms_wights[term] = 0.05
    # Check if the query has 'Geo' connection
    if len([term for term in org_tokens if term in location_queries['all']]) > 0:
        # if so, add words that can help find locations and search only within the body
        for term in location_queries['city and country']:
            if term not in tokens:
                tokens.append(term)
                terms_wights[term] = 0.05
        score_body = candidates_by_body(tokens,terms_wights, index_body)
        res = []
        for k,v in score_body.items():
            res.append((k,score_body.get(k, 0) + PR.get(k, 0)))
        # Sort by the relevance score
        res = sorted(res, key=lambda x: x[1], reverse=True)
        res = [(i[0], DT[i[0]]) for i in res][:10]
        return res

    # else, calculate scores for each part of the doc
    score_body = candidates_by_body(tokens,terms_wights, index_body)
    score_anchor = candidates_by_title_or_anchore(tokens, terms_wights, index_anchor)
    score_title = candidates_by_title_or_anchore(stem_tokens, terms_wights_with_stem, index_stem_title)
    res = []
    docs = np.unique(list(score_title.keys()) + list(score_body.keys()))
    for d in docs:
        # combine the score of each part with the matching weight
        res.append((d,w1*score_title.get(d,0)+w2*score_body.get(d,0)+w3*score_anchor.get(d,0)+w4*NPR.get(d,0)))
    # Sort by the relevance score
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = [(int(i[0]), DT[i[0]]) for i in res][:10]
    return res
