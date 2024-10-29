# ntlee1-midterm

I attempted to predict Amazon review scores using a classification model trained on a dataset of labeled reviews. The Amazon dataset contained both labeled and unlabeled reviews, where the labeled dataset was used for training and the other one was used for prediction. To approach this task, I employed numerous data preprocessing methods, feature engineering, model selection, model ensembling, and cross validation.

This dataset was imported using the Pandas library, a tool for handling and considering the size of the dataset (1 GB+), I sampled 10% of the labeled data for training purposes, which maximizes computational efficiency. I randomly selected the data, which allows the sampling to be representative of the whole dataset.

Once the data was loaded, text preprocessing was the next step in the process. The review text was cleaned, and stop words were filtered out, such as 'the', 'and', and 'in' in order to streamline the text data by retaining key words. The preprocessing function was applied to the training set using the tqdm library. 

Three new features were added to the set to assist with the model's predictions: text length, helpfulness ratio, and sentiment analysis. Text length was calculated by counting the number of each word in the processed review, giving me a basic measure of verbosity. Additionally, the helpfulness ratio allowed us to have an insight of the usefulness of each review, which was calculated by dividing the number of helpful votes by the total votes plus 1. Sentiment analysis was conducted using the TextBlob library which assigns a polarity score to each review, giving it a positive (1) and negative (-1) review. 

Additionally, a TF-IDF vectorization was applied to the textual data to convert it into numerical vectors based on word frequency, accounting for the frequency of words in the review, and its frequency across the whole dataset. This transformation enabled the model to capture more important words within reviews, making the data suitable for training; however, there was a problem of overfitting and computational complexity. Thus, I applied an SVD to the TF-IDF matrix reducing its dimensions while retaining its informative components.

Furthermore, I chose a diverse set of models, using XGBoost, LightGBM, Random forest, and Gradient Boosting. XGBoost and LightGDM are gradient-boosting models. Random Forest was used with its class weighting capability to address imbalance in review scores, while GradientBoosting, by Scikit-learn, provided further gradient boosting capabilities. This allows us to differentiate the labeled data and our prediction to guide the model to the best classifications possible. To address the probabilistic estimates of Random Forest and HistGradientBoosting, I employed classifiers which enhanced the model's probability outputs for more accurate classifications.

These models were combined using a voting ensemble method, which averages the predicted probabilities from each model to determine the final prediction. Soft voting allows every model to contribute to the final prediction based on its confidence. Cross-validation was conducted using a stratified K-Fold cross validation technique ensuring the distribution of classes remained consistent across folds. This technique reduces the likelihood of overfitting.

The evaluation of the model's performance was measured using accuracy, precision, and F1-score. Furthermore, I visualized the model's confusion matrix, providing insight to how well the model distinguishes between review scores. 

I had to make several assumptions during the model development. One was that the 10% of data used was representative of the overall training set; additionally, I assumed that the sentiment and helpfulness ratio would provide a useful signal for classification, the class imbalance aws a huge challenge. 



