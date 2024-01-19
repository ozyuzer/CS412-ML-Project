# CS412 Course Project - _Can we predict HW grades from ChatGPT interactions?_

_by_
_Alkım Özyüzer, Baturalp Arslan Kabadayı, Görkem Afşin, Deniz Özdemir, Nil Boyraz and Eren Özdil_

### Introduction
Welcome to our project repository! In this project, our aim was to develop a machine learning model to predict student homework grades based on their chat history with ChatGPT. We seeked to answer the question: Can we predict student homework grades based on their interactions with OpenAI's ChatGPT?
In order to answer this question, we came up with many features that can help to develop the model so that we can precisely predict the homework grades. However, some features that we have tried was not succesfully; either have no correlation or some features were not successfull to lower the mean squared error. Despite having some features that managed to make the model a little more precise we decided not to include them. Our decision was based on the logic; Correlation does not necesseraly imply causation, especially on not so large datasets.
We decided to include them as well so that the steps that we took and how we progress throughout the project can be examined clearly.
Here in this repository you can find all the code, data and the steps we took while developing our model. Below, you'll find detailed explanations on datasets and their preparation, features, model trainings and evalutations. 
## 1. Reading the Data

### Overview
In this project, we aim to estimate student homework grades based on their interactions with ChatGPT. The data provided includes HTML files of ChatGPT conversations and a CSV file containing corresponding student grades.

### Data Sources
- **ChatGPT Conversations**: A collection of HTML files, each representing a student's conversation history with ChatGPT.
- **Grades**: A CSV file named `scores.csv`, which contains student codes (to match with ChatGPT histories) and their respective grades.

### Steps for Data Preparation

#### a. Loading ChatGPT Conversations
- We use Python's `glob` and `BeautifulSoup` libraries to read and parse the HTML files.
- Each conversation is identified by a unique file code, which is used to map the conversation to the corresponding student's grade.
- We extract relevant parts of the conversation, focusing on both the user's prompts and ChatGPT's responses. Special attention is given to code snippets exchanged in the conversation.
- Conversations are filtered and cleaned, with a threshold set to exclude trivial interactions.

#### b. Processing Conversation Content
- Textual data from conversations are vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) to convert the qualitative data into a quantitative form suitable for analysis.
- Additional features are derived from the conversations, such as the average length of prompts/responses, the frequency of specific keywords, and sentiment analysis scores.

#### c. Loading and Preparing Grades Data
- The `scores.csv` file is read using the `pandas` library.
- We clean and preprocess this data, ensuring the student codes in the grades file match those in the conversation data. This step is crucial for accurately mapping grades to ChatGPT interactions.

#### d. Merging and Finalizing the Dataset
- The processed conversation data and grades are merged based on the student codes.
- This merged dataset is then used for further analysis, feature engineering, and model training.
---
## 2. Feature Engineering

### 2.1 Provided Features

Initially, we included a variety of keywords like "error", "thank", "no", "next" and others as features, extracted from student chats with GPT-based tools. These were intended to reflect different aspects of student engagement and understanding. However, after thorough analysis, we found that some keywords were not strong indicators of academic performance and were actually contributing to increased model error. Consequently, we streamlined our feature set by dropping these less impactful keywords, focusing on those that more effectively predicted student grades.

### 2.2 Custom Features

#### :x: a. Number of Apologize Words

In our predictive model for student grades, we initially considered the frequency of the word "apologize" in chats as a potential feature. The rationale behind this was that frequent apologies might indicate a lack of confidence or understanding, potentially correlating with lower academic performance. However, upon analyzing our dataset, we found that occurrences of "apologize" were infrequent and did not provide significant predictive value. In fact, including this feature led to an increase in model error, suggesting that it was not a strong indicator within the context of our specific dataset. Consequently, we decided to exclude the "apologize" count from our final model. This decision underscores our commitment to data-driven model refinement, where we continuously assess and validate each feature's relevance and impact on the model's accuracy and effectiveness.

#### ✅ b. Number of Code Lines & Cells

The 'Number of Code Lines & Cells' feature represents the total count of lines of code and individual code cells provided by ChatGPT within a conversation. We decided to add this feature because the amount of code provided by ChatGPT is an important measure for this project, considering the homework is done by using ChatGPT, and matching the number of code lines with student grades can contribute a lot to our model. The scatter plot below illustrates the relationship between this feature and student grades, implicating for which intervals students tend to get higher or lower grades. Adding this feature to our model caused a decent amount of decrease in measures of Mean Squared Error (MSE), indicating that number of code lines have a direct impact to the prediction abilities of our model.

![alt text](https://github.com/ozyuzer/cs412-ml-project/blob/main/plots/total_code_lines_scatter_plot.png)


#### ✅ c. Prompt Complexity Analysis

This custom feature implements a Flesch-Kincaid (FK) complexity analysis on user prompts. The FK grade level is a readability index that gauges the relative complexity of text in English. It calculates an approximate U.S. school grade level needed for comprehension, with higher values indicating more complex language. In the context of this analysis, each data point links the intricacy of language used by ChatGPT to a student's academic performance, providing insights into whether interactions with more sophisticated linguistic structures from the AI correlate with better or worse student grades. 
In the context of our project, we decided to keep this feature in our model because of its contribution to prediction. Though the below graph may not show a direct correlation between average FK grades and student grades, language complexity analysis is beneficial for the prediction of our model.

![alt text](https://github.com/ozyuzer/cs412-ml-project/blob/main/plots/average_fk_grade_scatter_plot.png)


#### ✅ d. Average Prompt per Question

The feature 'Average Prompt per Question' quantifies the average number of times students engaged with specific questions or prompts during their conversations with ChatGPT. After indentifying each prompt's cosine similarity to each question, the prompts are distributed to questions and average prompt per question is calculated. This information supplies insights to the model in a way that it shows how many prompts per each question results in better student grades. We decided to add this feature to our model because of its efficiency and its contribution to predicting performance of our model.


#### :x: e. Prompt Ratio
The main purpose of the feature “Prompt Ratio” was to capture the importance of the average number of prompts of each user by taking into account the prompt numbers of other users. This feature compares the average number of all the prompts with each user’s individual average number of prompts and demonstrates the relation. The idea behind this feature was that if the user’s average number of prompts differs from the total average number of promts, the probability that the user found the answer is getting lower in a similar ratio. When we examined the importance of this feature by introducing the graph of correlation of this feature with grade, we observed that the correlation is high. Although it seems that feature contributes to our model in a positive way, when we applied this feature to our model with other features, it increased the mean square error of the model. Thus, we didnt include this feature to our best model.

#### :x: f. Zero Sum
This feature calculates how many questions in total each user did not write any prompt for. We expected that there should be negative correlation between the number of zeros and the grade because each zero might mean that the user was not able to answer the question and recieve a grade for that question. (Here, number of zeros indicates the number of questions that user didnt write any prompt for). After assessing the significance of this particular feature by depicting its correlation graph with the grade, we noticed a substantial correlation. Despite the initial impression that this feature positively contributes to our model, its inclusion, when combined with other features, led to an increase in the mean square error of the model. Consequently, we decided to exclude this feature from our optimal model.


#### :x: g. Sentiment Anaylisis

We implemented sentiment analysis to enhance our prediction of student grades based on their chat interactions with GPT. We used TextBlob library for the implementation of sentiment analysis. This approach allows us to capture students' emotional responses, providing insights into their understanding and engagement with the subject matter. Positive sentiments often correlate with a better grasp of the content and a proactive learning attitude, while negative sentiments may indicate confusion or frustration. By integrating sentiment analysis, we aim to capture these subtle indicators of academic performance, making our predictive models more robust and nuanced. Although this tool is particularly valuable when combined with other analytical metrics, contributing to a comprehensive understanding of each student's learning experience, it was increasing the mean square error of our model. Hence, we did not included it as a feature in our best fit model.

#### :x: h. Message Length Variability

One of the features explored in our analysis was the "Message Length Variability," designed to capture the variability in the length of messages sent by users during their interactions with ChatGPT. This feature was computed as the standard deviation of the word count in each user's messages, with the intention of reflecting the consistency or variability in their query lengths. The hypothesis behind this feature was that a higher variability might indicate a user's struggle to articulate queries clearly or a varied range of query complexities, potentially correlating with their understanding of the material and, consequently, their grades. However, upon integration into our model, we observed an increase in the MSE. This outcome suggests that while intuitively appealing, the variability in message length did not positively contribute to predicting the homework grades in our specific dataset. This feature was ultimately not included in the final model but serves as an interesting avenue for understanding user interaction patterns.

#### :x: i. Engagement Score

This metric was designed to quantify a student's level of engagement in the conversation with ChatGPT, based on several parameters: the frequency of messages sent by the user, the diversity of topics covered (as inferred from the range of unique words used), the average length of the user's messages, and the responsiveness score (calculated as the ratio of user messages to ChatGPT responses). The idea was that higher engagement, indicated by frequent, varied, and substantial messages, might correlate with a deeper exploration of the subject matter, potentially reflecting on the student's understanding and effort. However, contrary to our expectations, incorporating the Engagement Score into our model resulted in an increased MSE. This outcome suggested that the measure of engagement, as defined by us, did not effectively predict the grades in the context of our dataset. The feature was thus excluded from the final model. This instance highlights the complexity of capturing and quantifying student engagement and its impact on academic performance in a machine learning context.


### 2.3 Merging Features and Target Variable
We merged the features and target variables for future use in model training and evaluation. To accomplish this, we initiated a data frame named "scores" by reading the CSV file "scores.csv," which contains the codes of ChatGPT and their corresponding grades. Following this, we conducted a sequence of left merge operations on the primary data frame, aligning it with additional data frames identified as important to the model (Question Mapping Scores, Number of Code Lines, Prompt Complexity Analysis, and Average Prompt per Question). This merging process was executed using the shared identifier "code" column for alignment. Then, we excluded features that exhibited negligible contribution to our model. At the final stage, to accomplish the objective of merging features and target variables, we merged the resulting data frame with the scores data frame based on common "grade" column.


## 3. Model Training and Evaluation

We have tried different models in order to get the best results. These models were:
    1. Decision Tree Regressor
    2. Random Forest Regressor
    3. XGBoost Regressor
    4. Neural Network
    5. Linear Regressor

Before doing any further feature selection on the data, XGBoost regressor gave the best results (MSE and R^2 score) which led us to progress with the XGBoost regressor model. The model with default parameters gave an MSE: 81.2 and R^2: 27.7% on the test data.

### 3.1. Feature Selection
In order to get the best results, we have wrote an automated script that would do the feature selection. First of all, we plotted feature importance of the XGB model. Then, we have trained different models, each using different subsets of the data. These subsets start from all the features and at each iteration, we drop the least important feature, by setting an importance threshold, to train an another model. We have done this iteration until there is only the most important feature. We have also set an early stopping criteria for these models not to overfit the training data. The results is as the following.

* Threshold: 0.0005, #Features: 32, MSE: 76.51
* Threshold: 0.0005, #Features: 31, MSE: 76.51
* Threshold: 0.0008, #Features: 30, MSE: 75.00
* Threshold: 0.0013, #Features: 29, MSE: 76.55
* Threshold: 0.0014, #Features: 28, MSE: 76.53
* Threshold: 0.0031, #Features: 27, MSE: 75.66
* Threshold: 0.0033, #Features: 26, MSE: 76.29
* Threshold: 0.0036, #Features: 25, MSE: 70.29
* Threshold: 0.0052, #Features: 24, MSE: 70.29
* Threshold: 0.0063, #Features: 23, MSE: 71.08
* Threshold: 0.0092, #Features: 22, MSE: 71.03
* Threshold: 0.0095, #Features: 21, MSE: 67.95
* Threshold: 0.0120, #Features: 20, MSE: 78.77
* Threshold: 0.0122, #Features: 19, MSE: 77.99
* Threshold: 0.0127, #Features: 18, MSE: 79.43
* Threshold: 0.0142, #Features: 17, MSE: 53.43
* Threshold: 0.0167, #Features: 16, MSE: 55.52
* Threshold: 0.0169, #Features: 15, MSE: 55.42
* Threshold: 0.0176, #Features: 14, MSE: 59.99
* Threshold: 0.0212, #Features: 13, MSE: 51.05
* Threshold: 0.0229, #Features: 12, MSE: 53.11
* Threshold: 0.0233, #Features: 11, MSE: 40.98
* Threshold: 0.0263, #Features: 10, MSE: 45.22
* Threshold: 0.0293, #Features: 9, MSE: 47.64
* Threshold: 0.0300, #Features: 8, MSE: 60.68
* Threshold: 0.0416, #Features: 7, MSE: 94.21
* Threshold: 0.0499, #Features: 6, MSE: 125.44
* Threshold: 0.0734, #Features: 5, MSE: 130.31
* Threshold: 0.0746, #Features: 4, MSE: 132.54
* Threshold: 0.0870, #Features: 3, MSE: 204.48
* Threshold: 0.0917, #Features: 2, MSE: 128.31
* Threshold: 0.2821, #Features: 1, MSE: 112.37

- Best threshold: **0.023293254896998405**
- Best MSE: **40.984**
- Best R2 Score: **63.494%**


As it can be seen in the results, the best MSE score is achieved by using the most important 11 features. Thus, our final model only includes these 11 features.

### 3.2. Tuning the Model
Finally, we tried the tune the model hyperparameters by using GridSearch. However, tuning the model caused MSE to increase; therefore, we decided not to tune the model. This may be because of the fact that the model overfits to the training dataset as our dataset is not large enough.

### 4. Team Contributions 
As a team, we regularly conducted meetings to determine the steps that we will take. Throughout this process, we committed to supporting each other as much as possible, ensuring that our progress was a collective effort. Although we distributed some tasks among the group; Creating the README file, updating the repository, adding features was collaborative effort. Despite the fact that everyone worked on almost every aspect of the project, below you can find specific work of our team members: 

  * Baturalp Arslan Kabadayı, 29287: Creating the feature Number of Code Lines & Cells, Developing the model, Model Training and Evaluation, Data Visualization and Creating Plots. 
  
  * Deniz Özdemir, 29194: Creating the feature Number of Apologize words, Sentimental Analysis, Data Visualization and Creating Heat Map to understand the data.
  
  * Alkım Özyüzer, 29263: Creating the feature Average Length of ChatgGPT responses per user, Understanding the Data and Data Preparations.
  
  * Nil Boyraz, 29251 : Creating the feature Zero Sum, Merging Features and Target Variable, Merging Scores with features.
  
  * Görkem Afşin, 29180: Creating the Number of Prompts per Question & Average Number of Prompts per Question feature, Prompt Complexity Analysis, Feature Selection.
  
  * Eren Özdil, 29409: Creating Message Length Variability feature, Engagement Score Analysis, Feature Analysis.
