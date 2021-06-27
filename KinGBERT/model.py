import spacy
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from KinGBERT.utils import extractor_topic_rank, extractor_topical_page_rank, extractor_single_rank, extractor_multipartite_rank



class KinGBERTExtractor:
    def __init__(
        self,
        extract_methods = ['TopicRank', "TopicalPageRank", 'SingleRank','MultipartiteRank'],
        top_k = 5,
        n_gram_range=(1, 2),
        spacy_model="en_core_web_sm",
        bert_model="sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    ):
        self.n_gram_range = n_gram_range
        self.nlp = spacy.load(spacy_model)
        self.model = AutoModel.from_pretrained(bert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.extract_methods = extract_methods
        self.top_k = top_k
        self.candidates = []

    def squash(self, value):
        if not torch.is_tensor(value):
            raise ValueError(f"unexpected `value` of type {value.__class__}")
        if value.ndim == 2:
            return value
        return value.mean(dim=1)


    def get_all_candidates(self, text):
        stop_words = "english"
        count = CountVectorizer(ngram_range=self.n_gram_range, stop_words=stop_words).fit([text])
        self.candidates = count.get_feature_names()
        if 'TopicRank' in self.extract_methods:
          self.candidates += extractor_topic_rank(text)
        if 'TopicalPageRank' in self.extract_methods:
          self.candidates += extractor_topical_page_rank(text)
        if 'SingleRank' in self.extract_methods:
          self.candidates += extractor_single_rank(text)
        if 'MultipartiteRank' in self.extract_methods:
          self.candidates += extractor_multipartite_rank(text)
        self.candidates = np.unique(self.candidates).tolist()

    def generate(self, text):
        text = text[:1000].lower()
        candidates = self.get_candidates(text)
        text_embedding = self.get_embedding(text)
        candidate_embeddings = self.get_embedding(candidates)
        distances = cosine_similarity(text_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0]][::-1]
        keywords_list = []
        for keyphrase in keywords:
          if (len([el.lower() for el in keyphrase.split(' ') if el.lower() in ' '.join(keywords_list).lower()])==0)&(len(keywords_list)<self.top_k):
            keywords_list.append(keyphrase)
        return keywords_list

    def get_candidates(self, text):
        nouns = self.get_nouns(text)
        self.get_all_candidates(text)
        candidates = list(filter(lambda candidate: candidate in nouns, self.candidates))
        return candidates

    def get_nouns(self, text):
        doc = self.nlp(text)
        nouns = set()
        for token in doc:
            if token.pos_ == "NOUN":
                nouns.add(token.text)
        noun_phrases = set(chunk.text.strip() for chunk in doc.noun_chunks)
        return nouns.union(noun_phrases)

    @torch.no_grad()
    def get_embedding(self, source):
        if isinstance(source, str):
            source = [source]
        tokens = self.tokenizer(source, padding=True, return_tensors="pt")
        outputs = self.model(**tokens, return_dict=True)
        embedding = self.parse_outputs(outputs)
        embedding = embedding.detach().numpy()
        return embedding


    def parse_outputs(self, outputs):
        value = None
        outputs_keys = outputs.keys()
        if len(outputs_keys) == 1:
            value = tuple(outputs.values())[0]
        else:
            for key in ["pooler_output", "last_hidden_state"]:
                if key in output_keys:
                    value = outputs[key]
                    break
        if value is None:
            raise RuntimeError("no matching BERT keys found for `outputs`")
        return self.squash(value)



if __name__ == '__main__':
    doc = """What is data science?
    Data science is a multidisciplinary approach to extracting actionable insights from the large and ever-increasing volumes of data collected and created by today’s organizations. 
    Data science encompasses preparing data for analysis and processing, performing advanced data analysis, and presenting the results to reveal patterns and enable stakeholders to draw informed conclusions.
    Data preparation can involve cleansing, aggregating, and manipulating it to be ready for specific types of processing. Analysis requires the development and use of algorithms, analysis and AI models. It’s driven by software that combs through data to find patterns within to transform these patterns into predictions that support business decision-making. The accuracy of these predictions must be validated through scientifically designed tests and experiments. And the results should be shared through the skillful use of data visualization tools that make it possible for anyone to see the patterns and understand trends."""

    extractor = KinGBERTExtractor()
    keywords = extractor.generate(doc)
    print(keywords)


