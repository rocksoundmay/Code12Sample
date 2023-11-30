#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Checking working directory
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

current_directory = os.getcwd()
print(current_directory)


# In[6]:


# Change working directory
new_directory_path = "/Users/kenzie/Downloads"
os.chdir(new_directory_path)

updated_dir = os.getcwd()
print(updated_dir)


# In[7]:


# File path
file_path = "Week14Assignment.txt"

try: 
    with open(file_path, "r") as file:
        content = file.read()
        print(content)
except FileNotFoundError:
    print("File '{file_path}' Not Found.")
except IOError:
    print("An error occured while reading the file.")
    
def read_patient_data(file_path):
    patient_data = []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            patient_info = line.strip().split(',')
            patient_data.append(list(map(int, patient_info)))
    return np.array(patient_data)


# In[8]:


# Function to calculate statistics
def calculate_statistics(data):
    num_readmitted = np.sum(data[:, 1])
    avg_satisfaction_scores = np.mean(data[:, 2:], axis=0)
    return num_readmitted, avg_satisfaction_scores


# In[28]:


# Function to perform logistic regression
def perform_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Display confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot logistic regression curve
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, color='black', zorder=20)
    X_test_range = np.linspace(X_test.min(), X_test.max(), 300)
    y_prob = model.predict_proba(X_test_range.reshape(-1, 1))[:, 1]
    plt.plot(X_test_range, y_prob, color='blue', linewidth=3)
    plt.xlabel('Overall Satisfaction Scores')
    plt.ylabel('Probability of Readmission')
    plt.title('Logistic Regression Curve')
    plt.show()
    # Calculate and display regression coefficients

    coefficient = model.coef_
    Intercept = model.intercept_
    print(f"\nRegression Coefficients: {coefficient} ")
    print(f"Intercept: {Intercept}")
    
    if coefficient <= 0.5:
        print("There is weak or no correlation between patient satisfaction scores and readmission.")
    else:
        print("There is a strong correlation between patient satisfaction scores and readmission.")


# In[29]:


# Main function
def main():
    file_path = 'Week14Assignment.txt'
    
    try:
        # Read patient data
        patient_data = read_patient_data(file_path)
        
        # Calculate statistics
        num_readmitted, avg_satisfaction_scores = calculate_statistics(patient_data)
        
        # Display statistics
        print(f"Number of Patients Readmitted: {num_readmitted}")
        print(f"Average Staff Satisfaction: {avg_satisfaction_scores[0]:.2f}")
        print(f"Average Cleanliness Satisfaction: {avg_satisfaction_scores[1]:.2f}")
        print(f"Average Food Satisfaction: {avg_satisfaction_scores[2]:.2f}")
        print(f"Average Comfort Satisfaction: {avg_satisfaction_scores[3]:.2f}")
        print(f"Average Communication Satisfaction: {avg_satisfaction_scores[4]:.2f}\n")
        
        # Prepare data for logistic regression
        X = patient_data[:, 2:].mean(axis=1).reshape(-1, 1)  # Overall satisfaction scores
        y = patient_data[:, 1]  # Readmission
        
        # Perform logistic regression
        perform_logistic_regression(X, y)
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




