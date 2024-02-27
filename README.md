# deep-learning-challenge
--------------------------
GITHUB DIRECT CODE LINK (AlphabetSoupCharity.ipynb): https://github.com/meera-patel01/deep-learning-challenge/blob/main/AlphabetSoupCharity.ipynb 

GITHUB DIRECT CODE LINK (AlphabetSoupCharity_Optimization.ipynb): https://github.com/meera-patel01/deep-learning-challenge/blob/main/AlphabetSoupCharity_Optimization.ipynb

## Analysis
- Overview: The purpose of this analysis is the nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
- Results: 
    - Data Preprocessing
        - Variables that are targets: IS_SUCCESSFUL
        - Variables that are features: IS_SUCCESSFUL
        - Variables that should be removed: For the initial analysis both the EIN and NAME columns are dropped, but in the optimized analysis only the EIN column is dropped.
    - Compiling, Training, and Evaluating
        - How many neurons, layers, and activation functions did you select for your neural network model, and why?
          ![image](https://github.com/meera-patel01/deep-learning-challenge/assets/80857225/39002961-5112-4a45-a1db-b0715055e7fc)
        - Were you able to achieve the target model performance? When dropping both the EIN and NAME columns the model accuracy is 72% but when only dropping the NAME column the target model performance was achieved, the model accuracy is 79%.
        - What steps did you take in your attempts to increase model performance? First I wanted to try only changing the number of columns dropped, I started only dropping EIN incorporated NAME into the analysis, and if that change didn't increase the model performance then I would drop more columns and change the number of layers to get the optimized solution. Fortunately, that simple change brought my model performance from 72% to 79%.
- Summary: Overall, the deep learning model performance accuracy was around 79% after optimization which is greater than the target.

## Background
From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

## Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

## Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

## Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    - Dropping more or fewer columns.
    - Creating more bins for rare occurrences in columns.
    - Increasing or decreasing the number of values for each bin.
    - Add more neurons to a hidden layer.
    - Add more hidden layers.
    - Use different activation functions for the hidden layers.
    - Add or reduce the number of epochs to the training regimen.

## Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

