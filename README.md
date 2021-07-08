# ReSt - A Deep Learning Approach for Italian Stereotype Detection in Social Media
## Introduction
The Hate Speech Detection (HaSpeeDe 2) task presented at Evalita 2020  was composed of the main task (hate speech detection) and two Pilot tasks (stereotype and nominal utterance detection). This project aims to investigate different models for solving the stereotype detection task. Our study includes different types of neural networks such as convolutional neural networks (CNNs), recurrent neural networks model (BiLSTM), and BiLSTM with a soft-attention module. We also evaluated a BERT model by using an Italian pre-trained BERT and then fine-tuned the entire model for our classification task. In our experiments, it emerged that the choice of model and the combination of features extracted from the deep models was important. Moreover, with Bert, we noticed how pre-trained models on large datasets can give a significant improvement when applied to other tasks.

This project was developed for the course of [Human Language Technologies](https://elearning.di.unipi.it/course/view.php?id=180) at the University of Pisa under the guide of [Prof. Giuseppe Attardi](http://pages.di.unipi.it/attardi/).

All the detalis can be found on the full report [here](https://github.com/alessandrocuda/ReSt/blob/main/report/HLT_Stereotype_detection_19_20.pdf).

## Table of Contents 
- [Usage](#usage)
- [Models](#models)
- [Results](#Results)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

## Usage
This code requires Python 3.8 or later, to download the repository:

`git clone https://github.com/alessandrocuda/ReSt`

Then you need to install the basic dependencies to run the project on your system:

```
cd ReSt
pip install -r requirements.txt
```

Download the [Italian Twitter Embeddings](http://www.italianlp.it/download-italian-twitter-embeddings/) and move to:

`!mv twitter128.bin results/model/word2vec`

and you are ready to go.

## Models 
All the models explored in this project are listed below and are all avaible as H5 tensorflow models in the [results](https://github.com/alessandrocuda/ReSt/tree/main/results/model) folder:
- **KCNN**, inspired by the [Kim’s model](https://arxiv.org/pdf/1408.5882.pdf)
![](https://www.researchgate.net/profile/Aleksander-Smywinski-Pohl/publication/331643881/figure/fig2/AS:735079986900994@1552268134872/Illustration-of-Kim-CNN-model-architecture.png)
- **D-KCNN**, a KCNN that combines text, PoS tags and all the extra features extracted in this project 
![](https://github.com/alessandrocuda/ReSt/blob/main/report/assest/double_cnn.png?raw=true)
- **D-BiLSTM**, follow the D-KCNN architecture but with two [BiLSTM](https://paperswithcode.com/method/bilstm)
![](https://github.com/alessandrocuda/ReSt/blob/main/report/assest/double_bilstm-2.png?raw=true)
- **A-BiLSTM**, concatenate the text and PoS tagging as input to a BiLSTM and to take advantage of all the features extracted by by the BiLSTM, we weighted each output with an [attention mechanism](https://aclanthology.org/W18-6226/).
![](https://github.com/alessandrocuda/ReSt/blob/main/report/assest/attention_bilstm2.png?raw=true)
- [BERT](https://arxiv.org/pdf/1810.04805.pdf) we used a cased pretrained bert model provided by [DBMZ](https://github.com/dbmdz/berts) and fine tuned to our task.



## Results
| Model  | Macro F1-score Test |
| ------------- | ------------- |
| BERT  | 0.737  |
| A-BiLSTM  |  0.722 |
| D-KCNN  |  0.715 |
| Baseline_SVC  |  0.714 |
| D-BiLSTM  |  0.703  |
| KCN  |  0.700  |
| Baseline_MFC  |  0.354  |

## Contributing
 
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

<!-- CONTACT -->
## Contact

Alessandro Cudazzo - [@alessandrocuda](https://twitter.com/alessandrocuda) - alessandro@cudazzo.com

Giulia Volpi - giuliavolpi25.93@gmail.com

Project Link: [https://github.com/alessandrocuda/ReSt](https://github.com/alessandrocuda/ReSt)

<!-- LICENSE -->
## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

This library is free software; you can redistribute it and/or modify it under
the terms of the MIT license.

- **[MIT license](LICENSE)**
- Copyright 2021 ©  <a href="https://alessandrocudazzo.it" target="_blank">Alessandro Cudazzo</a> - <a href="mailto:giuliavolpi25.93@gmail.com">Giulia Volpi</a>