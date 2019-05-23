# AI Lecture

## Table of Contents:
1. [Initial Notes](#initial-notes)
2. [Goal of Machine Learning](#goal-of-machine-learning)
3. [Linear Mapping](#linear-mappings)
4. [Linear and Ridge Regression](#linear-and-ridge-regression)
4. [Types of Losses](#types-of-losses)



## Initial Notes
- Key Terms:
    - Loss & Loss Function
    - Ground Truth
    - Accuracy
    - Label

- Loss: measures how much prediction deviates from ground truth
    - Useful since this determines how sensitive the model will be 

## Goal of machine learning:
1. To have low loss on **new, unseen data**
2. New data is uncertain -> Model it by drawing it from a probability distribution


- Why based on probability?
    - Building probability:
        - P(x,y) = p(x)p(y|x)
        - p(x,y) = p(y)p(x|y)

    - "*Noise*" will be present within input vector space
        - This comes in the form of unaccounted variables
        - Therefore, model input features and variables as coming from probability distributions instead of static

- Predictor
    - What makes a good predictor?
        - Generalize well
            - Average loss from datasets are low
                - Don't want to have everything low, since this would be probably overfitting

## Linear Mappings
- What does this represent?
    - The set of points which make up a dataset
    - **Affine mapping** modifies datapoints. For this, w is necessary.
    - Taking datapoints as vector x, and vector w as the orthogonal vector, for k dimensions, the datapoints exist within (k-1) dimensions. 
        - This is necessary for w to exist as a hyperplane orthogonal to the other points.
        - This hyperplane **must go through the origin**
        - Here, bias = 0
        - **Else: (shift in opposite direction)**
        ```
            - If bias < 0:
                hyperplane would be shifted in the direction of vector w
            - If bias > 0:
                hyperplane shifted in opposite direction of w
        ```
- How to solve for parameters?
    - Terms:
        - n: Size of dataset
        - D: dimensions, size of features
    - Rare case, refer to slides, transpose matrix and solve, such that (nxd)(dx1) = (nx1)
        - Then, (1xn)(nx1) will give a real number
        - Without bias, there is explicit solution
        - With bias, extend dimension -> D+1, with the D+1th dimension being the bias

## Linear and Ridge Regression
- Ridge regression is Linear with added penalty term on weights
- Since w (weights) help to fit the curve to the datapoints, overfitting may occur with large values of w.
    - To avoid this, add penalty to the eucidean length of w. *(Page 12)*
    - This penalty comes in the form of <span style="color:orange">*lambda*</span>, the hyperparameter 
        - Thus, tuning this hyperparameter helps reduce overfitting.


## Types of Losses
- **Hinge Loss**
    - Linear model for classfication
    - Preferable to 0-1 loss since it involes probabilities
- <span style="color:lightgreen;">WIP: will be expanded </span>

## Basis Functions
- Radial basis functions: 
    - Transform x -> *phi*(x)
    - Helps to measure how **similar** 2 datapoints are from each other
        - Also extends to collection of datapoints
    - w.*phi*(x) is the **weighted sum of similarities between x and the set points of P**

