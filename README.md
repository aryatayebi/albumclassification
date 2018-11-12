##### album-classification
Repository for album genre classification project

Our goal of this project is to explore different machine learning classification methods. To do this, we will attempt to classify an albums genre through its cover art. Albums of the same genre often have similar album art themes. For example, country albums often depict a full body shot of the artist, and contain a guitar, banjo, or other musical instrument related to country music.

Album covers will undoubtedly vary within certain genres, which will make it very difficult to deduce an unknown genre with near certainty. However, we hope that by training on key characteristics of album covers determined by computer vision feature extraction and multi classification algorithms, we can predict the genre with accuracy sufficiently greater than random.

### Data  
In order to train our models and test the accuracy of their predictions, we will need a data set containing identification of albums, the cover art for that album, and the genre of the album.
We will create our data set by selecting a subset of albums from last.fm, which contains a comprehensive database of albums and relevant album information. We will access the information in this database through the websites API [Album Data API]. Each album has a set of associated tags, which will contain the genre of the album. We will select about 10 genres, then pull the top hundred albums that are tagged with each genre. These are the albums we will train and test on. 
To access the album art for these selected albums , we will use an open source cover art archive [Album Cover API]. We can retrieve the album art directly from our code using the provided API. 

### Algortithms/Classifiers
Feature Extraction:
First, we will apply a variety of algorithms to obtain feature extraction from each training and test image. There are four main computer vision algorithms that we will use and compare. The algorithms are SURF, ORB, SIFT, and BRIEF which are mostly found in the OpenCV library and will be written in python. 
On our training data, we will extract the features from the album cover into a vector using one of these algorithms. If there are k features then the featureized extraction vector is Rk. Each training vector will correspond to a qualitative y-value that is the genre of the album such that y Rk.Therefore our training data will be transformed to a matrix X Rn x kwhere each row is a featurized sample with a total of n samples.
	On our testing data, we will perform the the same featurized extraction from each algorithm and store the results in a matrix F Rn x kwith the qualitative y-values stored in a vector t Rk.

Classification:
Next, we will run multiple classification algorithms on matrix X and vector  y. The algorithms we wish to test include: naive bayes, logistic regression, support vector machine, k nearest neighbors, and decision trees. All these algorithms are multi classification and will be written in python.

Prediction:
	Finally, we will apply each algorithm to matrix F and see if we classified the genre correctly using vector t.
  
### Conclusion
There will be a total 20 different types of approaches (4 different feature extraction algorithms and 5 different multi classification algorithms). However, this is subject to change depending on time.
From the predictions, we will report which pairs of feature extraction and multi classification algorithms best predicted the correct genre of an album cover.
