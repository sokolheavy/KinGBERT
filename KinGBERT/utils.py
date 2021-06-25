import nltk
import string
nltk.download('stopwords')
import pke
stoplist = list(string.punctuation)
stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']


def extractor_topic_rank(text):
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=text, language='en', normalization='stemming')
    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'}, stoplist=stoplist)
    extractor.candidate_filtering(maximum_word_number=2, minimum_word_size=3, minimum_length=4, only_alphanum=True)
    extractor.candidate_weighting(threshold=0.9, method='average', heuristic='frequent')
    topicrank_keywords = [phrase[0] for phrase in extractor.get_n_best(n=20, redundancy_removal=True)]
    return topicrank_keywords


def extractor_topical_page_rank(text):
    extractor = pke.unsupervised.TopicalPageRank()
    extractor.load_document(input=text, language='en', normalization='stemming')
    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'}, stoplist=stoplist)
    extractor.candidate_filtering(maximum_word_number=2, minimum_word_size=3, minimum_length=4, only_alphanum=True)
    extractor.candidate_weighting(window=10, pos={'NOUN', 'PROPN', 'ADJ'})
    topicalpagerank_keywords = [phrase[0] for phrase in extractor.get_n_best(n=30, redundancy_removal=True)]
    return topicalpagerank_keywords


def extractor_single_rank(text):
    extractor = pke.unsupervised.SingleRank()
    extractor.load_document(input=text, language='en', normalization='stemming')
    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})
    extractor.candidate_filtering(maximum_word_number=2, minimum_word_size=3, minimum_length=4, only_alphanum=True)
    extractor.candidate_weighting(window=10, pos={'NOUN', 'PROPN', 'ADJ'})
    singlerank_keywords = [phrase[0] for phrase in extractor.get_n_best(n=30, redundancy_removal=True)]
    return singlerank_keywords


def extractor_multipartite_rank(text):
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text, language='en', normalization='stemming')
    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})
    extractor.candidate_filtering(maximum_word_number=2, minimum_word_size=3, minimum_length=4, only_alphanum=True)
    extractor.candidate_weighting(alpha=1.1, threshold=0.74, method='average')
    multipartite_rank_keywords = [phrase[0] for phrase in extractor.get_n_best(n=30, redundancy_removal=True)]
    return multipartite_rank_keywords