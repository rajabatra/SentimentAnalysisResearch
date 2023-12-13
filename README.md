# SentimentAnalysisResearch
Researching Different Approaches and Models for conducting sentiment analysis on social media
In a digital world, sentiment analysis is an important tool that can be used to extract emotions from text. This paper explores several methods for deciphering sentiment within Twitter's 280 characters. Utilizing Kaggle’s Sentiment-140 dataset [cite], categorized into positive and negative tweets, our research employs advanced models to interpret Twitter sentiments: Transformers, LSTM’s, GPT 3.5 and GPT 4. We start by exploring the dataset and then detail the construction of the transformer and LSTM models. Our focus then shifts towards investigating the impact of in-context learning and prompt input sequences on foundation models.
System Design
A.	Dataset and preprocessing
The dataset utilized in our analysis was the kaggle sentiment 140 dataset [cite]. The dataset had 6 fields: target, ids, date, flag, user, and text. For the purpose of our research we filtered the data to contain the text, which was the text of a tweet, and target, which was the polarity of a tweet. As shown in figure 1, the dataset contained 1.6 million tweets that was evenly split between positive and negative tweets.

Figure 1. Distribution of Tweets

	The first step in processing our data was adjusting the labels. Each tweet had a target which is a 0 if it is negative and a 4 for positive. The adjustment made was redefining the positive label to have a value of 1 instead.
To process the text of the tweets, we utilized various techniques. For the transformer model, we attempted word level tokenization as well as subword tokenization. For the LSTM, we used word level tokenization and for the foundation models, we used the raw text.
The first technique we utilized was word level tokenization. To do this we processed each individual tweet so that all the URLs were replaced with ‘URL’, emojis with ‘emoji,’ and usernames with ‘user.’ We then lemmatized the text and removed stopwords from the text using the open source python package nltk [cite]. After, the remaining text was lowercase and filtered to only contain alphanumeric characters. For example, the following tweet “@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it.” was converted into “user url aww bummer shoulda got david carr third day.” Processing the 1.6 million tweets took a total time of 88 seconds using the google colab cpu. 
Once the text was filtered, we built a vocabulary for the text using the pytorch feature torchtext [cite]. With torchtext, a dictionary of the 20000 most common words in our dataset was created. Each word had a corresponding index. Using this vocabulary, the text in each tweet was mapped to its corresponding integer. The list of integers was padded so they were all the same length and ultimately converted to a pytorch tensor so that it could be stored in a pytorch DataLoader. 
   The second technique utilized was subword tokenization. Subword tokenization is the process of breaking down words into smaller units based on their frequency. This was done following a hugging face tutorial to build a wordpiece tokenizer from scratch [cite]. The first step was to normalize the text using the prebuilt BertPreTokenizer[cite]. Our tokenizer was then trained on the text file wikitext-2 [cite] to create a dictionary of 25000 tokens. We then encoded our twitter texts using this tokenizer so that the text was mapped to the new subwords we created. For example, the text “let’s test this tokenizer” was converted to ['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.', '[SEP]']. Using the encode function of the tokenizer, it took 144 seconds to process all of our tweets. The tokenized string was then mapped to the tokens corresponding integer. The encoded text was converted to a pytorch tensor and stored in a pytorch dataloader along with its label.    
C. Data Split
	To train the transformer and LSTM models, we used 20 percent of our data (320000 tweets). This smaller dataset was randomly selected while maintaining an even distribution of positive and negative tweets. From this subset, we used the sklearn library[cite] to split our dataset into 80 percent training, 10 percent validation, and 10 percent test. The encoded text and labels were stored in their respective pytorch dataloaders with 128 tweets per batch.
Model Approach
The next section will explore the three different model approaches we used. Our first approach was a custom transformer, followed by an LSTM, and ultimately foundation models.
A. 	Transformers
The first approach attempted was building a transformer model. Our model utilized an embedding layer, followed by a positional encoding layer, transformer layer, and linear layer. The transformer layer utilized the architecture from the paper “Attention is all you need”[cite] as shown in Figure 2. 

Figure 2. Transformer Block Diagram [cite]
Using our training and validation data, we tuned our model and ultimately settled on a model size with 5 heads, 2 layers, a hidden dimension of 80, and an embedding dimension of 40. In addition, during training, I implemented gradient clipping, weight bias, and added dropout layers. 
I first trained the model on my processed data that utilized word level tokenization. As seen in figure 3, the model was able to train, as the loss decreased and eventually converged.

Figure 3. Training Curve for Transformer with word level tokenization
The model took around 79 seconds per epoch to train. While the word level encoded text was able to train, it had a relatively low accuracy. The model achieved its highest validation accuracy of 61.12% and test accuracy of 61.01%. 
Our intuition suggested that there were too many words that were only seen a few times. As a result, it was difficult to train the model prompting us to attempt encoding our tweets with subword tokenization. In our second attempt, we utilized the same transformer model, but with the wordpiece tokenization.The training curve for our second attempt at training the transformer is depicted in Figure 4. This model was faster, taking 54 seconds per epoch. As well, we were able to achieve a much higher accuracy. Our validation data reached 80.81% accuracy and our test data had an accuracy of 78.08%.

Figure 4. Training Curve for Transformer with subword level tokenization
B.	Transformer
	The second approach we tested was building an LSTM, which is a recurrent neural network architecture. The model which is shown in Figure 5 uses an embedding layer, pytorch’s LSTM layer, a linear layer, and a sigmoid activation. The model then returns the output from the last layer to determine the sentiment. To train the model, we experimented with gradient clipping, dropout, different loss functions, and various model sizes. We ultimately settled on our LSTM size to have 2 layers, an embedding dimension of 256, and a hidden dimension of 512. 

Figure 5. LSTM Block Diagram 
	This model successfully trained as shown in Figure 6, and took around 40 seconds per epoch to train. As well, it received a peak validation accuracy of 74.6% and a test accuracy of 73.76%

Figure 6. Training Curve for LSTM model

