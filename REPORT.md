# Introduction

In recent years NLP has experienced a major breakthrough largley due to a bold approach taken recently by researcher Sebastian Ruder, a researcher at Deep Mind. As we have seen, one of the major breakthroughs in the application of Deep Learning approaches to NLP has been the learning of word embeddings by encoding each word in approaches such as word2vec and GloVe. Hence, words and sentences with similar semantic meaning take on similar representation in the high dimensional vector space of the word embeddings enabling to perform NLP tasks such as sentiment analysis with greater accuracy.

Ruder's, ULMFit, work takes insperation from the best practice of transfer learning from Computer Vision which proved to be very successful. The basic idea of transfer learning is to be able to perform a target task, with previous knowledge learned from a source task. In this work I perform transfer learning from the WikiText-103 dataset to a Twitter US Airline Sentiment dataset. The WikiText-103 dataset consists of 28,595 preprocessed Wikipedia articles whose contents sum up to a total of 103 million words. Luckily the weights learned for the source are opensourced permiting us to focus on the target task.


# The Problem
Our target task is to predict the sentiment of tweets associated with US Airlines. The Twitter US Airline Sentiment dataset contains 14,485 tweets concerning major US airlines where each tweet is one of three categories positive, negative or neutral. The intuition behind the ULMFit is that a language model shouldn't have to relearn for each task the basic foundations of a given language (i.e. what is a valid word). Therefore, ULMFit was trained on a large corpus in order to learn language and then I finetune the model for our specific task. The dataset can be found on the Kaggle competition site.

# The Goal

My goal in this work was to:

1. Achieve good accuracy in sentiment classificatoin.
2. Explore transfer learning with the usage of ULMFit model.
3. Attempt to use Multi-task learning to achieve better accuracy on sentiment classification.

What is Multi-task learning?
>wikipedia: Multi-task learning (MTL) is a subfield of machine learning in which multiple learning tasks are solved at the same time, while exploiting commonalities and differences across tasks.

Why perform Multi-task learning?
>wikipedia: This can result in improved learning efficiency and prediction accuracy for the task-specific models, when compared to training the models separately.

The confidence marker in my view is an indicator of how strong the sentiment label is. Therefore, I believe the task of learning the confidence should positively effect the task of sentiment analysis.


# Approach
The general framework for using ULMFit model involves three stages:

1. General-Domain LM Pretraining: LM is pretrained on a large general-domain corpus. At this stage the model learns the general features of the language.
2. Target Task LM Fine-Tuning: Adapt the trained model by fine-tuning it on the data of the target task. At this point, it has also learned task-specific features of the language, such as emojis.
3. Target Task Classifier: Use the pretrained LM in order to predict the final output which is a probability distribution over the sentiment labels positive, negative and neutral.

In my work I tested two approaches:

## Approach I
Following steps 1-3 as explained above.

## Approach II
Perform steps 1-3 

Perform steps 1-3 as explained above. Then after saving the network, take the additional step of learning an additional task of confidence of sentiment from the Twitter US Airline Sentiment dataset. Finally repeat step 3 above. The motivation to this approach was to simply test the effects of a Multi-task learning approach while training.


In order to see and validate that the ULMFit model approach is an approach worth while I used scikit learn models as a baselines.

# Results

The baseline algorithm with RandomForestClassifieris achieved an accuracy of 0.7636612021857924. Training ULMfit initially achieved an accuracy of 0.788934. 

Afterwhich the hypothesis of Multi-task learning by training on confidence was tested and rejected since it achieved an accuracy of 0.635246 vs continueing to train on the sentiment label which achieved an accuracy of 0.810109 (for the same number of episodes).

Finally the ULMfit model continued to learn the sentiment label and after two epochs achieved its best accuracy of 0.817395.


# Conclusion

The pure ULMfit approach surpassed the baseline and Multi-task learning approache's accuracies. Acheiving such accuracy attests to the success in learning the sentiment of the airline tweets. 

Note that the Multi-task approach had a detrimental effect on the Neural Network causing it to decrease in performance on the original task of sentiment analysis. A further work in may be to explore different approaches of Multi-task learning such as adding an additional output in training the neural network which would predict the confidence.