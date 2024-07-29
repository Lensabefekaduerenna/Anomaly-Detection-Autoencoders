# Anomaly-Detection-Autoencoders
I will train an autoencoder to detect anomalies on a dataset that contains 5,000 Electrocardiograms, each with 140 data points. I will use a simplified version of the dataset, where each example has been labeled either 0 (corresponding to an abnormal rhythm), or 1 (corresponding to a normal rhythm). I'm interested in identifying the abnormal rhythms.

This is a labeled dataset, so one could phrase this as a supervised learning project. The goal of this project is to illustrate anomaly detection concepts I can apply to larger datasets, where I do not have labels available (for example, if I had many thousands of normal rhythms, and only a small number of abnormal rhythms). An autoencoder is trained to minimize reconstruction error. I will train an autoencoder on the normal rhythms only, then use it to reconstruct all the data. The hypothesis is that the abnormal rhythms will have higher reconstruction error. I will then classify a rhythm as an anomaly if the reconstruction error surpasses a fixed threshold.
I will show this step by step.
<h2>Import TensorFlow and other libraries</h2>
<img width="650" alt="image" src="https://github.com/user-attachments/assets/14492acd-347e-4784-9e84-fb48d891bfc9">
<h2>Load ECG data</h2>
<img width="664" alt="image" src="https://github.com/user-attachments/assets/0cba4e3e-a46e-43a2-8724-f4da68b0ee50">
<img width="641" alt="image" src="https://github.com/user-attachments/assets/7a3a7ffd-7b04-4794-b945-93cb22c5d24e">

<h2>Normalize the data to [0,1] to improve training accuracy.</h2>
<img width="667" alt="image" src="https://github.com/user-attachments/assets/6ac0861e-5490-4d3c-b72d-9fd4cfcd8d91">
<h2>I will train the autoencoder using only the normal rhythms, which are labeled in this dataset as 1. Separate the normal rhythms from the abnormal rhythms.</h2>
<img width="643" alt="image" src="https://github.com/user-attachments/assets/e02a7c07-bfd1-4b0e-ad4e-43dbbb865e6b">

<h2>Plot a normal ECG</h2>
<img width="682" alt="image" src="https://github.com/user-attachments/assets/9388fa28-c47d-4aab-9246-8fa88aa0864b">
<h2>Plot an anomolus ECG</h2>
<img width="617" alt="image" src="https://github.com/user-attachments/assets/3124bb02-5df1-4494-aca9-dd973ce29ae0">
<h2>Build the model</h2>
<b>After training and evaluating the example model, I try modifying the size and number of layers to build an understanding for autoencoder architectures. I found out changing the size of the embedding (the smallest layer) can produce interesting and better results.</b>
<img width="663" alt="image" src="https://github.com/user-attachments/assets/7b32f749-26f9-42af-9ac5-68d914c3e771">
<h2>Train the model</h2>
<b>The autoencoder is trained using only the normal ECGs, but is evaluated using the full test set.</b>
<img width="653" alt="image" src="https://github.com/user-attachments/assets/7546271d-6bf7-4457-ad81-47363dbc7b90">
<img width="653" alt="image" src="https://github.com/user-attachments/assets/c26d071f-c963-43bc-ad2f-a849f4f9b45d">
<h2>Evaluate the training</h2>

<b>Soon we'll classify an ECG as anomalous if the reconstruction error is greater than one standard deviation from the normal training examples. First, let's plot a normal ECG from the training set, the reconstruction after it's encoded and decoded by the autoencoder, and the reconstruction error.</b>

<img width="632" alt="image" src="https://github.com/user-attachments/assets/127d3167-1389-43b6-8d3f-e824d1bb8108">

<h2>Then I'll create a similar plot, this time for an anomalous test</h2>
<img width="640" alt="image" src="https://github.com/user-attachments/assets/85cc9455-df98-41b0-8630-42ed8dd7d2bc">
<h2>Detect anomalies</h2>
<b>Detect anomalies by calculating whether the reconstruction loss is greater than a fixed threshold. I will calculate the mean average error for normal examples from the training set, then classify future examples as anomalous if the reconstruction error is higher than one standard deviation from the training set.</b>

<b>Plot the reconstruction error on normal ECGs from the training set</b>
<img width="611" alt="image" src="https://github.com/user-attachments/assets/40218179-f4ba-4a3d-9eb1-10496c0725f7">

<h2>Choose a threshold value that is one standard deviations above the mean.</h2>
<img width="608" alt="image" src="https://github.com/user-attachments/assets/cc48386d-a665-48e1-b985-650d7cff4991">

<b>If we examine the recontruction error for the anomalous examples in the test set, we'll notice most have greater reconstruction error than the threshold. By varing the threshold, we can adjust the "precision" and "recall" of our classifier.</b>

<img width="581" alt="image" src="https://github.com/user-attachments/assets/ee9337f7-4cf4-48fd-bfd0-d87a25450f2d">

<h2>Classify an ECG as an anomaly if the reconstruction error is greater than the threshold.</h2>

<img width="617" alt="image" src="https://github.com/user-attachments/assets/be2ed2b4-09b1-4a77-b983-1c03a29060ca">
<h2>ROC and AUC Metrics</h2>
<b>We've created a fairly accurate model for anomaly detection but our accuracy is highly dependant on the threshold we select. What if we wanted to evaluate how different thresholds impact our true positive and false positive rates? We'll enter Receiver Operating Characteristic (ROC) plots.This metric allows us to visualize the tradeoff between predicting anomalies as normal (false positives) and predicting normal data as an anomaly (false negative). we remember that normal rhythms are labeled as 1 in this dataset.</b>
<img width="596" alt="image" src="https://github.com/user-attachments/assets/14aeb256-68cd-4f1d-88cf-df95ceb42b7d">
<img width="570" alt="image" src="https://github.com/user-attachments/assets/7a37ef03-c9b6-4b5e-9bcd-fbfa322cfffc">

<b>Since our model does a great job in differentiating normal rythms from abnormal ones it seems easy to pick the threshold that would give us the high true positive rate (TPR) and low false positive rate (FPR) that is at the 'knee' of the curve. However, in some cases there may be an application constraint that requires a specific TPR or FPR, in which case we would have to move off of the 'knee' and sacrifice overall accuracy. In this case we might rather have false alarms than miss a potentially dangerous rythm.</b>

<b>If we wanted to compare the performance of models without factoring in the threshold? Simply comparing the accuracy won't work since that depends on the threshold we pick and that won't have the same impact across models.Instead we can measure the area under the curve (AUC) in the ROC plot. One way to interpret the AUC metric is as the probability that the model ranks a random positive example more highly than a random negative example.</b>

<img width="613" alt="image" src="https://github.com/user-attachments/assets/f9591a41-2f97-4516-b985-0b58948eb9b7">
<h2>In conclution the AUC is a useful metic for comparison as it is threshold invariant and scale invariant</h2>











