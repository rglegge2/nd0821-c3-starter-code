# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Robert Legge created this model.
* The model is a RandomForestClassifier with hyperparameters selected with default hyperparameters.
* The version of scikit-learn used is 1.3.0.

## Intended Use
* This model should be used to predict whether an individual makes at least $50K per year based on attributes found within census data.

## Training Data
* Data was obtained from the [UC Irvine Matchine Learning Repository](https://archive.ics.uci.edu/dataset/20/census+income).
* The target class used was `salary` with values of either `>50K` or `<=50K`.
* The dataset contains a total of 32561 rows, and an 80-20 split was used to create train and test datasets with stratification done on the target variable.
* To convert the data to a usable format, a OneHotEncoder and LabelBinarizer were used on the categorical features and target class labels, respectively.

## Evaluation Data
* The evaluation data used was 20% of the original dataset, which is 6513 rows.
* The same processing was performed on both the train and test datasets.

## Metrics
* The model was evaluated using Precision, Recall, and F1-Score.
* Performance metrics are as follows:
    * Precision: 0.72796 
    * Recall: 0.61097 
    * F1-Score: 0.66436
    * Confusion Matrix: [[4587  358] [ 610  958]]
* For performance metrics on each of the features, please see [slice_output.txt](starter/slice_output.txt).

## Ethical Considerations
* The population sample for this dataset is not an accurate representation of the overall population, and therefore would not produce accurate results outside the scope of this dataset.
* This can be seen when looking at the lack of balance in certain features such as `sex`, where the sample size of males is twice that of females.
* Included factors such as `race` and `native_country` also introduce bias when we see that the number of individuals whose race is White make up ~85% of the sample population and those whose native country is the United States make up ~90%.

## Caveats and Recommendations
* For a true analysis of salary within the United States, consider a more balanced dataset that includes more equal representation across all features included within this dataset
* It is also worth noting that the data itself is from the 1994 Census database, and therefore is not representative of the current population. The most recent Census would be a better option.
* It may also be worth expanding the salary ranges from just <=50K or >50K to include additional labels to identify different levels of wealth within the country.