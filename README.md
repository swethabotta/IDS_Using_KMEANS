**Intrusion Detection using K-Means Clustering**
This project explores the application of the K-Means clustering algorithm for anomaly detection in network traffic data, with the goal of building an effective intrusion detection system (IDS).

**Overview**
Intrusion detection systems play a crucial role in identifying and preventing cyber threats in today's digital landscape. Traditional signature-based detection methods often fall short in detecting novel or evolving attacks. This project investigates the use of unsupervised machine learning, specifically the K-Means clustering algorithm, to detect anomalies in network traffic data, which could indicate potential intrusions or cyber attacks.

**Dataset**
The project utilizes the NSL-KDD dataset, a widely-used benchmark dataset for intrusion detection systems. This dataset contains network traffic data collected from a simulated military network environment, including features such as duration, protocol type, service, bytes transferred, flags, error rates, and more.

**Implementation**
The implementation follows these key steps:
1.Data Preprocessing: Relevant features are selected, and the dataset is preprocessed, including binary encoding of the 'attack' column, one-hot encoding of categorical features, and normalization of numerical features.
2.Hyperparameter Tuning: The RandomizedSearch method is used to find the optimal hyperparameters for the K-Means algorithm, using the silhouette score as the scoring metric.
3.Model Training and Evaluation: The K-Means model is trained on the preprocessed data using the tuned hyperparameters. The model's performance is evaluated using various metrics, including confusion matrix, accuracy, precision, recall, and F1-score.
4.Dimensionality Reduction: Principal Component Analysis (PCA) is explored as a dimensionality reduction technique to improve the model's performance and computational efficiency.
5.Comparison with Other Algorithms: The performance of K-Means is compared with other algorithms, such as K-Nearest Neighbors (KNN), Random Forest, and Isolation Forest, to assess their suitability for intrusion detection tasks.

**Results and Discussion**
![image](https://github.com/swethabotta/IDS_Using_KMEANS/assets/169571533/7bba323f-aab8-4c37-9c33-18e899249fcb)
True Positive (TP) = 10971; meaning 10971 positive class data points were correctly classified by the model
True Negative (TN) = 496; meaning 496 negative class data points were correctly classified by the model
False Positive (FP) = 2451; meaning 2451 negative class data points were incorrectly classified as belonging to the positive class by the model
False Negative (FN) = 11277; meaning 11277 positive class data points were incorrectly classified as belonging to the negative class by the model


The project provides a detailed analysis of the K-Means algorithm's performance for intrusion detection, including its strengths, limitations, and potential improvements. The results highlight the challenges of using K-Means for this task, such as its sensitivity to initial centroid positions, assumption of spherical clusters, and difficulty in handling unbalanced and high-dimensional data.

**Future Work**
Based on the insights gained from this project, several potential improvements and alternative approaches are suggested:
Exploring ensemble methods that combine K-Means with other anomaly detection algorithms to leverage the strengths of multiple models.
Developing methods to enhance interpretability and explainability of the learned cluster representations, which is crucial for intrusion detection systems.
Investigating adaptive and online learning techniques to adapt to concept drift and evolving attack patterns.
Handling high-dimensional data more effectively by using sparse data representations or dimensionality reduction techniques.
Evaluating supervised learning algorithms (e.g., Random Forest, SVM, Neural Networks) and anomaly detection techniques (e.g., Isolation Forest, One-Class SVM, Autoencoders) that may be better suited for intrusion detection tasks.
