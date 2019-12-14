<!----- Conversion time: 1.252 seconds.


Using this Markdown file:

1. Cut and paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β17
* Sat Dec 14 2019 07:32:56 GMT-0800 (PST)
* Source doc: https://docs.google.com/a/my.uri.edu/open?id=125gg2PkSOgZ3qPrF8yozadG9Rsl1Yr7TZcUvPzCoU0o
----->


**Identifying Individuals By Their Cell Phone Accelerometer Data**


            Ethan Kulman				 Isaac Pontarelli	

                               University of Rhode Island                              University of Rhode Island	


              ethan_kulman@my.uri.edu	 	                ipontarelli@my.uri.edu

**Abstract**

Mass data collection has become commonplace with the widespread adoption of smartphones and smartwatches. This data could be improperly used, however, there are many positive use cases that this data may have. The ability to determine an individual based off of the collection of routine data, such as motion data, may be useful to research related to drug rehabilitation compliance, law enforcement, user authentication, etc.

The purpose for this project is to determine whether or not it is possible to accurately identify an individual based off of accelerometer data collected by that individual's cell phone sensors in a non-controlled environment. Several studies have utilized other data sets to perform similar experiments. In this paper, we identify useful tactics employed in previous research to help design our own data pre-processing, and machine learning models, and end by comparing our data set and results to those studies we attempted to improve upon. Our best resulting model for accuracy was a Random Forest, which had an accuracy of 96.875%.



1. **Introduction**

There have been few studies that focus on person recognition using smartphone accelerometer data. On the other hand, there have been many studies that focus on Human Activity Recognition (HAR), which aims to identify what an individual is doing based off of the data collected by a bodily worn sensor. Much of the current research related to studying cell phone accelerometer data has been to determine its usefulness for Human Activity Recognition. On the other hand, person recognition from cell phone accelerometer data has both positive and negative implications, and therefore should be investigated. It has been noted in previous studies that one motivation for studying person recognition is for improving the security authentication for cell phones [2]. At the same time, person recognition poses privacy threats that can be exploited by governments and corporations who could misuse our data.

For this study, we have decided to build upon the work of previous research in order to study the potential for person recognition with a data set that mimics the real world. In order to have a modern reference to compare our results to, we decided to mimic part of the study design done by Singha et. al. [2]. In section 2 we will describe our data set. Next in section 3 we will discuss the related work which we used as a reference to compare our results to. In section 4 we detail our pre-processing technique. In section 5 we go over our model selection. In section 6 we discuss our results, and section 7 we will analyze our results in the discussion and conclusion. Finally in section 7, and 8  respectively we will go over our challenges and future work. 



2. **Data Set**

 The Wireless Sensor Data Mining (WISDM) Lab at Fordham University works to develop open source datasets for primarily researching Human Activity Recognition. In 2010 the WISDM lab published a paper titled, “Cell Phone-Based Biometric Identification” by Kwapisz et. al.. This study looked at how well the machine learning models at that time could perform person recognition using cell phone accelerometer data in a real world environment. This  This study was able to accurately identify a test set of 10 users at around 80% accuracy, and a test set of 36 users at around a 70% accuracy [4]. Since that time, the WISDM Lab has created several data sets from participants who have consented to have their cell phone sensor data mined for research purposes. One of these data sets are from the ActiTracker app created by the WISDM Lab [3]. This application is available only in the Android store. Over 500 users have downloaded the app and consented to having their cell phone x, y, and z axis accelerometer data mined for research purposes, generating a total of 2,980,765 entries. All of this data has been collected by unique users in a non-controlled environment, and this data has been labeled either by the user or by the WISDM lab as one of 6 different activities [3]. The 6 labeled activities in this dataset are: Walking, Jogging, Stairs, Sitting, Standing, Lying Down. The WISDM lab expects its users of the Actitracker app to have their phone positioned in their front pocket [3].

The data is collected at a 20hz sampling rate. Each data point is comprised of the user id, a timestamp, activity label, and the raw x, y, and z accelerometer values. 

There is also a demographic file included in the database with height, sex, age, weight and whether or not the user has a leg injury. Unfortunately this information is not complete and there are many users which have incomplete or missing entries

One such paper that has used the Actitracker data set is the “Human Activity Recognition Using LSTM-RNN Deep Neural Network Architecture” by Pienaar et. al. This study looked at the application of using a Long Short Term Memory-Recurrent Neural Network architecture to perform Human Activity Recognition. This study performed little preprocessing besides windowing the data set, and was able to achieve a 94% accuracy of predicting a particular activity label [1]. The results of this study show that the Actitracker dataset has valuable enough data to perform accurate human activity recognition. 

Since this dataset has real world data from a non-controlled environment, and it has been successfully used in previous studies, we decided to choose this data set to develop and test our machine learning models.



3. **Related Work**

There is one particular study performed more recently which we attempted to use as a baseline to compare our results to. This particular study is titled, “Person Recognition using Smartphones’ Accelerometer Data”, written by Singha et. al. in 2017 [2]. This paper describes a controlled environment study where 10 participants had a Samsung Galaxy J-1 cell phone record x, y, and z axis accelerometer data while the subjects performed trials where they walked [2]. The pre-processing steps utilized in this paper included windowing the data to two second windows with a 50% overlap, and performing feature transformations for both the time and frequency domains of the data [2]. Ultimately, the dataset they used to develop and test their machine learning models included 31 features, where 15 of those are from the time domain and 16 are from the frequency domain [2]. The researchers then tested four different models on the processed data, and utilized k-fold cross validation with 10 splits to determine each models testing accuracy. These four models are a Random Forest, Decision Tree, Logistic Regression, and a Support Vector Machine. Ultimately, this study found that the Random Forest classifier performed best, and accurately classified 96.79% of the test examples.



4. **Pre-Processing**

We attempted to follow the same pre-processing techniques done by Singha et. al. [2].  First, we windowed the x, y, z axis accelerometer data in 2 second windows with a 50% overlap. The data in this data set was collected at a rate of 20 hertz. Therefore each window of data is 40 data points, where each data point consists of x, y, and z accelerometer axis values. Each window was then transformed into 19 features in the time domain, and 6 features from the frequency domain, which partially mimics Singha et. al. [2]. We did not use all of the features listed in the study by Singha et. al. because there were some ambiguities about which features were only derived from the time domain, and which features were only derived from the frequency domain. Here is a list of the following features we used: 

(i) The mean of each of the x, y, and z axes in the time domain. (ii) The mean of each of the x, y, and z axes in the frequency domain.   (iii) The median of the x, y, and z axes in the time domain. (iv) The median of the x, y, and z axes in the frequency domain. (v) The magnitude of the entire window in the time domain. (vi) The correlation between the x and z, and the correlation between the y and z axes in the time domain. (vii)  The number of peaks for the x, y, and z axes in the time domain. (viii) The average number of peaks between all axes in the time domain. (viii) The average width between peaks in the x, y, and z axes in the time domain. 

The exact way to calculate each of these values is detailed further in paper written by Singha et. al. [2]. Ultimately, we decided to use only 10 users from the data set who had at least 10 minutes of usable data. The data from these 10 users is what was used to train, test, and validate the machine learning models that were selected.



5. **Model Selection**

Our model selection, and validation process was chosen to match precisely the same library implementations of the four models tested in the study by Singha et. al. [2]. These four models are: Random Forest, Decision Tree, Logistic Regression, and a Support Vector Machine. These models were implemented and validated using the sklearn libraries. In order to validate these models, we used k-fold cross validation with 10 splits. 



6. **Results**

The results from all four models demonstrates that our pre-processing methods and model configuration allowed the models to accurately perform person recognition. These results are from a final 70%/30% split between our training and test set. This split was done after verifying all of our models with k-fold cross validation. The results discussed from here on are the classification results from running our models on that test set containing 30% of the overall processed data. The results vary between the four models, and this is detailed below in Figure 1.


<table>
  <tr>
   <td>Model
   </td>
   <td>Precision
   </td>
   <td>AUC
   </td>
   <td>Recall
   </td>
  </tr>
  <tr>
   <td>RF
   </td>
   <td>0.9642
   </td>
   <td>0.9797
   </td>
   <td>0.9635
   </td>
  </tr>
  <tr>
   <td>DT
   </td>
   <td>0.9457
   </td>
   <td>0.9693
   </td>
   <td>0.9447
   </td>
  </tr>
  <tr>
   <td>LR
   </td>
   <td>0.9146
   </td>
   <td>0.9519
   </td>
   <td>0.9135
   </td>
  </tr>
  <tr>
   <td>SVM
   </td>
   <td>0.8877
   </td>
   <td>0.9259
   </td>
   <td>0.8666
   </td>
  </tr>
</table>


Figure 1. Precision, Area Under Curve, and Recall for each model tested.

The accuracy of the models obtained in our study compared to Singha et. al. are better for all models except for the Decision Tree. At the same time, the area under the curve and recall in our study for the Random Forest and Decision Tree were worse than those obtained by Singha et. al. 

In Figure 2, we detail the success of each model on a user by user basis. This figure shows that certain users were classified perfectly by the Random Forest, and those same users were classified less accurately by the Support Vector Machine. These are very similar findings to the results obtained by Singha et. al.. 




<table>
  <tr>
   <td>User
   </td>
   <td>RF
   </td>
   <td>DT
   </td>
   <td>LR
   </td>
   <td>SVM
   </td>
  </tr>
  <tr>
   <td>585
   </td>
   <td>0.9208
   </td>
   <td>0.9583
   </td>
   <td>0.9192
   </td>
   <td>0.8974
   </td>
  </tr>
  <tr>
   <td>602
   </td>
   <td>0.9691
   </td>
   <td>0.9029
   </td>
   <td>0.8980
   </td>
   <td>0.9684
   </td>
  </tr>
  <tr>
   <td>603
   </td>
   <td>0.9891
   </td>
   <td>0.9787
   </td>
   <td>0.9524
   </td>
   <td>0.9556
   </td>
  </tr>
  <tr>
   <td>608
   </td>
   <td>0.9600
   </td>
   <td>0.9406
   </td>
   <td>0.9400
   </td>
   <td>0.9625
   </td>
  </tr>
  <tr>
   <td>641
   </td>
   <td>0.9109
   </td>
   <td>0.8431
   </td>
   <td>0.8265
   </td>
   <td>0.8111
   </td>
  </tr>
  <tr>
   <td>648
   </td>
   <td>0.9362
   </td>
   <td>0.9222
   </td>
   <td>0.8830
   </td>
   <td>0.9326
   </td>
  </tr>
  <tr>
   <td>669
   </td>
   <td>0.9677
   </td>
   <td>0.9785
   </td>
   <td>0.9778
   </td>
   <td>0.5796
   </td>
  </tr>
  <tr>
   <td>675
   </td>
   <td>1.0000
   </td>
   <td>1.0000
   </td>
   <td>1.0000
   </td>
   <td>1.0000
   </td>
  </tr>
  <tr>
   <td>688
   </td>
   <td>1.0000
   </td>
   <td>0.9897
   </td>
   <td>0.9126
   </td>
   <td>0.9100
   </td>
  </tr>
  <tr>
   <td>1750
   </td>
   <td>0.9889
   </td>
   <td>0.9432
   </td>
   <td>0.8367
   </td>
   <td>0.8605
   </td>
  </tr>
</table>


Figure 2. Precision per person across all models

One notable finding of our results is how successful logistic regression was in classifying the test examples. This model performed much better in our research compared to the results obtained by Singha et. al.. One reason for this may be the configuration we gave the sklearn logistic regression model. We increase the number of epochs to 1,200 since we were having issues with convergence. This led to longer runtimes for the algorithm, but overall much better accuracy.

One particular reason that a Random Forest in our case may far outperform a support vector machine is because the decision boundary between users may be very close. This creates issues for a support vector machine since it will find it difficult to draw a clear boundary between classes. This would provide reason for the support vector machine’s lesser performance in both our results and the results by Singha et. al.



7. **Discussion And Conclusion**

The differences in our results compared to Singha et. al. can be accounted for by several different reasons. The first and most obvious reason would be that we used a real world data set. We expected to get different results using a real world data set, however, these were not the exact results we anticipated. Our overall success metrics were much better than we initially thought they would be. The WISDM lab stated in their research that they expect the users of the Actitracker app to have their phones in their front pocket [3]. However, user error may cause an individual's accelerometer data to be easily distinguished from all others. This was mediated by selecting only users with a large amount of data, standardizing the data using the sklearn standard scaler, and getting rid of homogenous data. The fact that we were able to obtain an accuracy of 96.875% on our 30% split test set indicates that there are clear differences between users accelerometer data that can be recognized by the Random Forest model.

 Another reason for the differences in our results may be that we did not use the exact configuration of sklearn’s machine learning models as Singha et. al.. All of the code we used was written using only the sklearn documentation. We did not have access to any of the code used in the study performed by Singha et. al..

Lastly, some of the differences in our results compared to Singha et. al. may be due to the fact that we did not use every feature that they did. We chose to only include the features that we could be certain we were calculating correctly. We were uncertain about properly calculating some of the methods described in the research by Singha et. al.. We thought that the best policy to avoid introducing incorrect features was to not include them at all. 

We performed another validation test utilizing 30 users with at least 5 minutes of data. After training, the Random Forest was still able to accurately classify above 90% of all test examples it was given. Not only do these results support our main findings, but they also show a dramatic improvement compared to the 2010 WISDM lab study performed by Kwapisz et. al. [4].

Overall, we can conclude that person recognition from cell phone accelerometer data can be accurately done with real world data. This provides support to the generalizability of the research done by Singha et. al.. It is clear from our results that the Random Forest machine learning model performs best for this task. 

**Challenges**

The first challenge that was encountered when performing this study was the inconsistency of the data set. When attempting to process the data set, we uncovered extraneous characters, empty columns, and users with homogenous data. Uncovering and removing these inconsistencies allowed us to start developing and testing our model properly. 

Another challenge we encountered was the lack of demographic information that was available in the WISDM Actitracker data set. Without sufficient demographic data, it’s difficult to say that our results would be generalizable enough to be able to distinguish between individuals of the same height, weight, and gender. We believe that the ability to accurately classify a dataset of 30 users with at least 5 minutes of data, such as the test described in the discussion section, helps support that our findings may be generalizable without having prior knowledge of an individual's demographic information. 



8. **Future Work**

A natural next step from the research we performed would be to test our findings with a larger data set. This would allow us to further test the generalizability of our results. In this future study, it would also be beneficial to have a clear set of features and definitions for those features. This would allow us to better compare our future results to past studies. 



References



1. Pienaar, Schalk Wilhelm, and Reza Malekian. “Human Activity Recognition Using LSTM-RNN Deep Neural Network Architecture.” _2019 IEEE 2nd Wireless Africa Conference (WAC)_, 2019, doi:10.1109/africa.2019.8843403.
2. Singha, T.B., Nath, R.K., & Narsimhadhan, A.V. (2017). Person Recognition using Smartphones' Accelerometer Data. ArXiv, abs/1711.04689.
3. Weiss, Gary M., et al. “Actitracker: A Smartphone-Based Activity Recognition System for Improving Health and Well-Being.” _2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA)_, 2016, doi:10.1109/dsaa.2016.89.
4. Kwapisz, Jennifer R., et al. “Cell Phone-Based Biometric Identification.” _2010 Fourth IEEE International Conference on Biometrics: Theory, Applications and Systems (BTAS)_, 2010, doi:10.1109/btas.2010.5634532.

<!-- Docs to Markdown version 1.0β17 -->
