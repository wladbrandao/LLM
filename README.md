# Large Language Models
A language model is a probability distribution over words or sequences of words \[[46](#Jurafsky:2023@book)\]. Given a sequence of words of length $m$, a language model assigns a probability $P(w_{1},\ldots ,w_{m})$ to the whole sequence based on training data from text corpora.

In large language models (LLMs), deep neural networks with a huge number of parameters (billions or trillions) are trained on massive datasets of unlabelled text to assign probabilities.

## Language Modeling
Language modeling is the process of the development of models that are able to predict the next word in a sequence given the words that precede it \[[35](#Ponte:1998@SIGIR)\]. Language modeling is a root problem for a large range of natural language processing tasks. Practically, language models are used as a basis for more sophisticated models for tasks that require language understanding. Therefore, they are crucial components in real world applications such as machine translation and automatic speech recognition, playing a central role in information retrieval, natural language processing, artificial intelligence, and machine learning research \[[16](#Goldberg:2017@book), [25](#Manning:2008@book)\].
Considering the word prediction strategy, one can categorize language models into three distinct groups: probabilistic, topic, and neural models.

### Probabilistic Models
Probabilistic models, also known as probabilistic language models or statistical language models, are probabilistic distributions over sequences or words that are able to predict the next word in a sequence given the words that precede it \[[35](#Ponte:1998@SIGIR)\]. A probabilistic model learns the probability of word occurrence based on examples of text. Simpler models may look at a context of a short sequence of words, whereas complex models may work at large sequences. Formally, a probabilistic model can be represented by a conditional probability of the next word given all the previous ones \[[3](Bengio:2003@JMLR)\], since


$$P(w_{1}^T) = \prod _{{i=1}}^{T}P(w_{t}\mid w_{1}^{t-1})$$

where $w_t$ is the t-th word, and a sequence $w_{i}^{j} = (w_i, w_{i+1},\ldots,w_{j-1}, w_j)$. Such models have already been used to address many natural language problems, such as speech recognition, language translation and information retrieval. The most popular models used for this purpose are the Bag of Words (BoW) and n-gram models.

#### BoW
BoW is an acronym for *Bag of Words*, a probabilistic language model where all words in a text are mutually independent. It is called a "bag" of words, since text structure, the order and distance of words are discarded, only concerning with whether known words occur, not where, next or close to. The intuition is that similar text share similar words, regardless of their structure. In this model, the probability of each word only depends on that word's own probability in the text. Formally, the sum of the probabilities of each word in the text vocabulary is one:
$$\sum_{{i=1}}^{T}P(w_{i}) = 1$$
BoW is a very simple and easy to understand model, hence it has been used on several information retrieval and natural language problems. Nevertheless, it has some limitations. First, the size of the vocabulary impacts the sparsity of the text representation. Accordingly, sparse representations are harder to compute (space and time complexity) and to leverage the available information in such a large representational space. Second, discarding word correlations, such as the order and distance of words, the model ignores context and therefore the meaning of the words in text. Semantics can help to address several information retrieval and natural language problems, by incorporating important correlation between words such as polysemy, synonyms and hyponyms.

An usual text representation with BoW represents a text as a $T$-dimensional vector of reals numbers, where $T$ is the size of the text vocabulary, i.e., the set of unique words in the text, and real numbers are word metrics extracted from the text. For instance, considering the text *\"the who is the band\"*, their vocabulary is composed by four words: *\"band\"*, *\"is\"*, *\"the\"* and *\"who\"*. Therefore, one represents the text using BoW as $[1.0, 1.0, 2.0, 1.0]$, where each dimension corresponds to a word in the vocabulary and the real numbers are the frequency of the word in the text. Analogously, for a collection of texts one have a larger vocabulary and therefore larger vectors representing each text of the collection. The representation of large text collections commonly culminates in the curse of dimensionality \[[45](#Chen:2009@book)\]. Is such phenomena, when the dimensionality increases, the volume of the space increases so fast that the data (vectors) become sparse. Course dimensionality is a fundamental problem, making language modeling and learning tasks too difficult \[[3](Bengio:2003@JMLR)\].

A widely used approach to address the high dimensionality - that leads to sparsity - problem is managing vocabulary.
Sparse vectors require more memory and computational resources therefore decreasing the size of the vocabulary improves text processing efficiency. There are simple but effective techniques that can be used as a preliminary strategy, such as ignoring case, punctuation and stop words, fix misspelled words and stemming \[[25](Manning:2008@book)\]. Although these techniques do not definitively solve the sparsity problem, they offer a way to lower the computational cost at the price of a semantically more restrictive representation of the original text, which may be useful in some learning, natural language processing, and information retrieval tasks.

In addition to word frequency, one can also use different metrics for the real-valued vectors. TF-IDF is a very popular word weighting schema in information retrieval, providing a numerical statistic that is intended to reflect how important a word is to a text in a collection. Particularly, the TF-IDF value of a word increases proportionally to the word frequency in the text (term frequency) and decreases proportionally to the number of distinct texts it occurs in the collection (inverted document frequency). There are different variants proposed in literature for accounting TF and IDF metrics \[[25](Manning:2008@book)\], and one of the most used variants defines $TF = \log(1+f_{{t,d}})$, where $f_{{t,d}}$ is the frequency of a word $t$ in a text $d$, and $IDF = \log ({N}/{n_{t}})$, where $N$ is the number of texts in the collection and $n_{t}$ is number of distinct texts where the word $t$ occurs in the collection.

For instance, for a collection composed of three texts $T_1 = $ *\"the who is the band\"*, $T2 = $ *\"who is the band\"* and $T3 = $ *\"the band who plays the who\"*, one have a vocabulary composed of five words: *\"band\"*, *\"is\"*, *\"plays\"*, *\"the\"* and *\"who\"*. Therefore, one represents the texts using BoW with TF-IDF as $T_1 = [1.0, 1.0, 2.0, 1.0]$, $T_2 = [1.0, 1.0, 2.0, 1.0]$ and $T_3 = [1.0, 1.0, 2.0, 1.0]$.

#### n-grams

To solve the limitations of [BoW model](#BoW), various advanced text representation strategies have been proposed. The n-gram statistical language models [2] were proposed to capture the term correlation within document. However, the exponentially increasing data dimension with the increase of N limits the application of N-gram models.

In fact, the BoW model is also known as the unigram model (n-gram model with $n=1$).

### Topic Models
Topic models are statistical models for discovering latent topics from text

#### LSA
The Latent Semantic Indexing (LSI) [3] was proposed to reduce the polysemy and synonym problems. At the same time, LSI can also represent the semantics of text documents through the linear combination of terms, which is computed by the Singular Value Decomposition (SVD). However, the high complexity of SVD [1] make LSI seldom used in real IR tasks. In addition, some external resources such as the Wordnet and Wikipedia are recently used for solving the polysemy and synonym problems in text representation. Since the effectiveness of these external resources for text representation can only be learned in research papers and there still has no evidence to show their power in real IR applications, this article will not introduce their details.

#### PLSA
Motivated by the LSI, the Probabilistic Latent Semantic Indexing (PLSI) [6] is also proposed for representing the semantics of text documents. However, it is still limited by the computation complexity due to the increasing scale of Web data. As a summary, though various approaches have been proposed for solving the limitations of BOW, the BOW with TFIDF term weighting schema is still one of the most commonly used text representation strategies in real IR applications.

#### PDA
Motivated by the LSI, the Probabilistic Latent Semantic Indexing (PLSI) [6] is also proposed for representing the semantics of text documents. However, it is still limited by the computation complexity due to the increasing scale of Web data. As a summary, though various approaches have been proposed for solving the limitations of BOW, the BOW with TFIDF term weighting schema is still one of the most commonly used text representation strategies in real IR applications.

### Neural Models
Neural models use neural network to provide continuous representations or embeddings of words

# References

<a name="Arora:2017@ICLR"></a>\[[1][1]\] Sanjeev Arora, Yingyu Liang, and Tengyu Ma. An efficient framework for learning sentence representations. In Proceedings of the 5th International Conference on Learning Representations, ICLR’17, April 2017. <!--- SIF -->

<a name="Baker:1962@JACM"></a>\[[2][2]\] Frank B. Baker. Information retrieval based upon latent class analysis. Journal of the ACM, 9(4):512–521, October 1962. <!--- LSA -->

<a name="Bengio:2003@JMLR"></a>\[[3][3]\] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. A neural probabilistic language model. The Journal of Machine Learning Research, 3:1137–1155, March 2003. <!--- NPLM -->

<a name="Berwick:2015@book"></a>\[[4][4]\] Robert C. Berwick and Noam Chomsky. Why Only Us: Language and Evolution. MIT Press, 2015.

<a name="Blei:2003@JMLR"></a>\[[5][5]\] David M. Blei, Andrew Y. Ng, and Michael I. Jordan. Latent dirichlet allocation. The Journal of Machine Learning Research, 3:993–1022, March 2003. <!--- LDA -->

<a name="Bojanowski:2017@TACL"></a>\[[6][6]\] Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov.  Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics, 5:135–146, June 2017. <!--- FastText -->

<a name="Brown:1992@CL"></a>\[[7][7]\] Peter F. Brown, Peter V. deSouza, Robert L. Mercer, Vincent J. Della Pietra, and Jenifer C. Lai. Class-based n-gram models of natural language. Computational Linguistics, 18(4):467–479, December 1992. <!--- N-gram -->

<a name="Cer:2018@EMNLP"></a>\[[8][8]\] Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Brian Strope, and Ray Kurzweil. Universal sentence encoder for English. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP’18, pages 169–174, October 2018. <!--- USE -->

<a name="Conneau:2018@CoRR"></a>\[[9][9]\] Alexis Conneau and Douwe Kiela. SentEval: An evaluation toolkit for universal sentence representations. In Proceedings of the 19th International Conference on Language Resources and Evaluation, LREC’18, pages 1699–1704, May 2018. <!--- Evaluation -->

<a name="Conneau:2017@EMNLP"></a>\[[10][10]\] Alexis Conneau, Douwe Kiela, Holger Schwenk, Loïc Barrault, and Antoine Bordes. Supervised learning of universal sentence representations from natural language inference data. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP’17, pages 1532–1543, September 2017. <!--- InferSent -->

<a name="Deerwester:1990@JASIS"></a>\[[11][11]\] Scott Deerwester, Susan T. Dumais, George W. Furnas, Thomas K. Landauer, and Richard Harshman. Indexing by latent semantic analysis. Journal of the American Society for Information Science, 41(6):391–407, 1990. <!--- LSI -->

<a name="Firth:1935@TPS"></a>\[[12][12]\] John Rupert Firth.  The technique of semantics. Transactions of the Philological Society, 34(1):36–73, 1935. <!--- Foundations -->

<a name="Firth:1957@book"></a>\[[13][13]\] John Rupert Firth. A synopsis of linguistic theory 1930-1955. In Studies in Linguistic Analysist, pages 1–32. 1957. <!--- Foundations -->

<a name="Gabrial:2007@book"></a>\[[14][14]\] Brian Gabrial.  History of writing technologies. In Handbook of Research on Writing: History, Society, School, Individual, Text, pages 27–39. July 2007.

<a name="Glasersfeld:1977@book"></a>\[[15][15]\] Ernst Von Glasersfeld. Linguistic communication: Theory and definition. In Language Learning by a Chimpanzee, pages 55-71. Academic Press, 1977.

<a name="Goldberg:2017@book"></a>\[[16][16]\] Yoav Goldberg, and Graeme Hirst. Neural Network Methods in Natural Language Processing. Morgan \& Claypool Publishers, 2017.

<a name="Harris:1954@WORD"></a>\[[17][17]\] Zellig S. Harris. Distributional structure. WORD, 10:146–162, 1954. <!--- BoW -->

<a name="Hill:2016@NAACL"></a>\[[18][18]\] Felix Hill, Kyunghyun Cho, and Anna Korhonen. Learning distributed representations of sentences from unlabelled data. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT’16, pages 1367–1377, June 2016. <!--- FastSent -->

<a name="Iyyer:2015@ACL"></a>\[[19][19]\] Mohit Iyyer, Varun Manjunatha, Jordan Boyd-Graber, and Hal Daumé III. Deep unordered composition rivals syntactic methods for text classification. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics, ACL’15, pages 1681–1691, July 2015. <!--- DAN -->

<a name="Jernite:2017@CoRR"></a>\[[20][20]\] Yacine Jernite, Samuel R. Bowman, and David Sontag. Discourse-based objectives for fast unsupervised sentence representation learning. CoRR, 1705.00557, 2017. <!--- DiscSent -->

<a name="Kiros:2018@EMNLP"></a>\[[21][21]\] Jamie Ryan Kiros and William Chan. InferLite: Simple universal sentence representations from natural language inference data. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP’18, pages 4868–4874, October 2018. <!--- InferLite -->

<a name="Kiros:2015@NIPS"></a>\[[22][22]\] Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. Skip-Thought vectors. In Proceedings of the 29th Conference on Neural Information Processing Systems, NIPS’15, pages 1532–1543, December 2015. <!--- Skip-Thought -->

<a name="Levy:2015@TACL"></a>\[[23][23]\] Omer Levy, Yoav Goldberg, and Ido Dagan. Improving distributional similarity with lessons learned from word embeddings. Transactions of the Association for Computational Linguistics, 2015.

<a name="Logeswaran:2018@ICLR"></a>\[[24][24]\] Lajanugen Logeswaran and Honglak Lee. An efficient framework for learning sentence representations. In Proceedings of the 6th International Conference on Learning Representations, ICLR’18, April 2018. <!--- Quick-Thought -->

<a name="Manning:2008@book"></a>\[[25][25]\] Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. Introduction to Information Retrieval. Cambridge University Press, 1 edition, July 2008.

<a name="Manning:1999@book"></a>\[[26][26]\] Christopher D. Manning and Hinrich Schütze. Foundations of Statistical Natural Language Processing. MIT Press, 1 edition, May 1999. <!--- N-gram -->

<a name="McCann:2017@CoRR"></a>\[[27][27]\] Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. Learned in translation: Contextualized word vectors. CoRR, 1708.00107, 2017. <!--- CoVe -->

<a name="Mikolov:2013@CoRR"></a>\[[28][28]\] Tomas Mikolov, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013. <!--- Skip-gram -->

<a name="Mikolov:2013@NIPS"></a>\[[29][29]\] Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. Distributed representations of words and phrases and their compositionality. In Proceedings of the 26th International Conference on Neural Information Processing Systems, NIPS’13, December 2013.

<a name="Nie:2017@CoRR"></a>\[[30][30]\] Allen Nie, Erin D. Bennett, and Noah D. Goodman. DisSent: Sentence representation learning from explicit discourse relations. CoRR, 1710.04334, 2017. <!--- DisSent -->

<a name="Pagliardini:2018@NAACL"></a>\[[31][31]\] Matteo Pagliardini, Prakhar Gupta, and Martin Jaggi. Unsupervised learning of sentence embeddings using compositional n-gram features. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLP’18, pages 528–540, June 2018. <!--- Sent2Vec -->

<a name="Pennington:2014@EMNLP"></a>\[[32][32]\] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. GloVe: Global vectors for word representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP’14, pages 1532–1543, October 2014. <!--- GloVe -->

<a name="Perone:2018@CoRR"></a>\[[33][33]\] Christian S. Perone, Roberto Silveira, and Thomas S. Paula. Evaluation of sentence embeddings in downstream and linguistic probing tasks. CoRR, 1806.06259, 2018. <!--- Evaluation -->

<a name="Peters:2018@NAACL"></a>\[[34][34]\] Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLP’18, pages 2227–2237, June 2018. <!--- ELMo -->

<a name="Ponte:1998@SIGIR"></a>\[[35][35]\] Jay M. Ponte and W. Bruce Croft. A language modeling approach to information retrieval. In Proceedingsof the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR’98, pages 275–281, 1998.

<a name="Salton:1975@CACM"></a>\[[36][36]\] Gerald Salton, Andrew Wong, and Yang Chung-Shu. A vector space model for automatic indexing. Communications of the ACM, 18(11):613–620, November 1975. <!--- BoW -->

<a name="Scott-Phillips:2013@JRSI"></a>\[[37][37]\] Thomas C. Scott-Phillips and Richard A. Blythe. Why is combinatorial communication rare in the natural world, and why is language an exception to this trend? Journal of the Royal Society Interface, 10(88), 2013.

<a name="Subramanian:2018@ICLR"></a>\[[38][38]\] Sandeep Subramanian, Adam Trischler, Yoshua Bengio, and Christopher J. Pal. Learning general purpose distributed sentence representations via large scale multi-task learning. In Proceedings of the 6th International Conference on Learning Representations, ICLR’18, April 2018. <!--- GenSen -->

<a name="Vaswani:2017@NIPS"></a>\[[39][39]\] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of the 31st Conference on Neural Information Processing Systems, NIPS’17, December 2017. <!--- Transformer -->

<a name="Weaver:1955@book"></a>\[[40][40]\] Warren Weaver. Translation. In Machine Translation of Languages: Fourteen Essays, pages 15–23. 1955. <!--- Foundations -->

<a name="Ledell:2018@AAAI"></a>\[[41][41]\] Ledell Wu, Adam Fisch, Sumit Chopra, Keith Adams, Antoine Bordes, and Jason Weston. StarSpace: Embed all the things. In Proceedings of the 32nd AAAI Conference on Artificial Intelligence, AAAI’18, pages 5569–5577, February 2018. <!--- StarSpace -->

<a name="Yan:2009@book"></a>\[[42][42]\] Jun Yan. Text representation. In Encyclopedia of Database Systems, pages 3069–3072. 2009.

<a name="Gulli:2005@WWW"></a>\[[43][43]\] Antonio Gulli and Alessio Signorini. The indexable Web is more than 11.5 billion pages. In Proceedings of the 14th International Conference on World Wide Web, WWW’15, pages 902–903, May 2015.

<a name="vandenBosch:2016@SCIENTOMETRICS"></a>\[[44][44]\] Antal van den Bosch, Toine Bogers, and Maurice de Kunder. Estimating search engine index size variability: a 9-year longitudinal study. Scientometrics, 107(2):839–856, May 2016.

<a name="Chen:2009@book"></a>\[[45][45]\] Lei Chen. Curse of dimensionality. In Encyclopedia of Database Systems, pages 545–546. Springer, 2009.

<a name="Jurafsky:2023@book"></a>\[[46][46]\] Dan Jurafsky, and James H. Martin. Speech and Language Processing. 3 Edition, 2023.

[1]: https://arxiv.org/abs/1803.02893
[2]: http://doi.org/10.1145/321138.321148
[3]: http://dl.acm.org/citation.cfm?id=944919.944966
[4]: https://doi.org/10.7551/mitpress/9780262034241.001.0001
[5]: http://dl.acm.org/citation.cfm?id=944919.944937
[6]: http://dx.doi.org/10.1162/tacl_a_00051
[7]: http://dl.acm.org/citation.cfm?id=176313.176316
[8]: http://dx.doi.org/10.18653/v1/D18-2029
[9]: https://www.aclweb.org/anthology/L18-1269
[10]: http://dx.doi.org/10.18653/v1/d17-1070
[11]: http://dx.doi.org/10.1002/%28SICI%291097-4571%28199009%2941%3A6%3C391%3A%3AAID-ASI1%3E3.0.CO%3B2-9
[12]: https://doi.org/10.1111/j.1467-968X.1935.tb01254.x
[13]: http://cs.brown.edu/courses/csci2952d/readings/lecture1-firth.pdf
[14]: https://doi.org/10.4324/9781410616470.ch2
[15]: https://doi.org/10.1016/B978-0-12-601850-9.50009-9
[16]: https://doi.org/10.2200/S00762ED1V01Y201703HLT037
[17]: http://dx.doi.org/10.1080/00437956.1954.11659520
[18]: http://dx.doi.org/10.18653/v1/n16-1162
[19]: https://www.aclweb.org/anthology/P15-1162
[20]: https://arxiv.org/abs/1705.00557
[21]: http://dx.doi.org/10.18653/v1/D18-1524
[22]: https://arxiv.org/abs/1506.06726
[23]: http://dx.doi.org/10.1162/tacl_a_00134
[24]: https://arxiv.org/abs/1803.02893
[25]: https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf
[26]: https://nlp.stanford.edu/fsnlp/
[27]: https://arxiv.org/abs/1708.00107
[28]: https://arxiv.org/abs/1301.3781
[29]: https://arxiv.org/abs/1310.4546
[30]: https://arxiv.org/abs/1710.04334
[31]: http://dx.doi.org/10.18653/v1/n18-1049
[32]: http://www.aclweb.org/anthology/D14-1162
[33]: https://arxiv.org/abs/1806.06259
[34]: http://dx.doi.org/10.18653/v1/n18-1202
[35]: http://doi.org/10.1145/290941.291008
[36]: http://doi.acm.org/10.1145/361219.361220
[37]: http://doi.org/10.1098/rsif.2013.0520
[38]: https://arxiv.org/abs/1804.00079
[39]: https://arxiv.org/abs/1706.03762
[40]: http://www.mt-archive.info/Weaver-1949.pdf
[41]: https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16998/16114
[42]: https://doi.org/10.1007/978-0-387-39940-9_420
[43]: http://doi.org/10.1145/1062745.1062789
[44]: https://doi.org/10.1007/s11192-016-1863-z
[45]: https://doi.org/10.1007/978-0-387-39940-9_133
[46]: https://web.stanford.edu/~jurafsky/slp3/3.pdf
