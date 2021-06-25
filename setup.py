from setuptools import setup

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name="KinGBERT",
    version="0.0.1",
    description="Keywords extractor with Graph and BERT methods",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/sokolheavy/KinGBERT",
    author="OlenaSokol",
    author_email="sokolooo1996@gmail.com",
    keywords="Keywords extractions Graph BERT",
    license="MIT",
    packages=["KinGBERT"],
    install_requires=["nltk==3.6.2",
                        "spacy==3.0.6",
                        "scikit-learn==0.24.2",
                        "transformers==4.8.0",
                        "torch==1.9.0",
                      ],

    include_package_data=True,
    zip_safe=False,
)