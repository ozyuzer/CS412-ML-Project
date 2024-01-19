# CS412 Course Project - Can we predict HW grades from ChatGPT interactions? 
## 1. Reading the Data
## 2. Feature Engineering
###  - 2.1 Provided Features
  In our grade prediction project, we initially included a variety of keywords like "error", "thank", "no", "next" and others as features, extracted from student chats with GPT-based tools. These were intended to reflect different aspects of student engagement and understanding. However, after thorough analysis, we found that some keywords were not strong indicators of academic performance and were actually contributing to increased model error. Consequently, we streamlined our feature set by dropping these less impactful keywords, focusing on those that more effectively predicted student grades.
  
###  - 2.2 Custom Features
#### :x: a. Number of Apologize Words
  In our predictive model for student grades, we initially considered the frequency of the word "apologize" in chats as a potential feature. The rationale behind this was that frequent apologies might indicate a lack of confidence or understanding, potentially correlating with lower academic performance. However, upon analyzing our dataset, we found that occurrences of "apologize" were infrequent and did not provide significant predictive value. In fact, including this feature led to an increase in model error, suggesting that it was not a strong indicator within the context of our specific dataset. Consequently, we decided to exclude the "apologize" count from our final model. This decision underscores our commitment to data-driven model refinement, where we continuously assess and validate each feature's relevance and impact on the model's accuracy and effectiveness.

#### ✅ b. Number of Code Lines & Cells

#### ✅ c. Prompt Complexity Analysis

#### ✅ d. Average Prompt per Question

#### :x: e. Sentiment Anaylisis 
  We implemented sentiment analysis to enhance our prediction of student grades based on their chat interactions with GPT. We used TextBlob library for the implementation of sentiment analysis. This approach allows us to capture students' emotional responses, providing insights into their understanding and engagement with the subject matter. Positive sentiments often correlate with a better grasp of the content and a proactive learning attitude, while negative sentiments may indicate confusion or frustration. By integrating sentiment analysis, we aim to capture these subtle indicators of academic performance, making our predictive models more robust and nuanced. Although this tool is particularly valuable when combined with other analytical metrics, contributing to a comprehensive understanding of each student's learning experience, it was increasing the mean square error of our model. Hence, we did not included it as a feature in our best fit model. 
#### :x: f. Message Length Variability
#### :x: g. Engagement Score
  ### 2.3 Merging Features and Target Variable
## 3. Model Training and Evaluation
