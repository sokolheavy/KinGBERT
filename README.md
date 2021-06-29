# KinGBERT

KinGBERT(Keywords in Graph with BERT) is a minimal keyword extraction library with graph methods to extract keywords. Use Sentence-BERT embedding for founding the most significant keywords.

## Installation

KinGBERT is available on PyPI.

```
pip install KinGBERT
```

To clone this repository, run

```
git clone https://github.com/sokolheavy/KinGBERT.git
```

## How to use

We use the `KinGBERTExtractor` class, which can be configured to generate keywords from text.
```python
text = """What is data science?
    Data science is a multidisciplinary approach to extracting actionable insights from the large and ever-increasing volumes of data collected and created by today’s organizations. 
    Data science encompasses preparing data for analysis and processing, performing advanced data analysis, and presenting the results to reveal patterns and enable stakeholders to draw informed conclusions.
    Data preparation can involve cleansing, aggregating, and manipulating it to be ready for specific types of processing. Analysis requires the development and use of algorithms, analysis and AI models. It’s driven by software that combs through data to find patterns within to transform these patterns into predictions that support business decision-making. The accuracy of these predictions must be validated through scientifically designed tests and experiments. And the results should be shared through the skillful use of data visualization tools that make it possible for anyone to see the patterns and understand trends."""
```
Just extract 5 keywords from the text.

```python
>>> from KinGBERT import KinGBERTExtractor
>>> extractor = KinGBERTExtractor(top_k=5)
>>> keywords = extractor.generate(text)
>>> print(keywords)
['data science', 'insights', 'analysis', 'experiments', 'algorithms']
```
