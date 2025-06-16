import pandas as pd
import numpy as np

# Sample student names and subjects for demonstration
student_names = ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Helen', 'Ivan', 'Julia']
subjects = ['Math', 'Physics', 'Chemistry', 'Biology', 'English']

# Generate DataFrame with random scores between 50 and 100
np.random.seed(42)  # For reproducibility in testing
data = {
    'Name': np.random.choice(student_names, size=10),
    'Subject': np.random.choice(subjects, size=10),
    'Score': np.random.randint(50, 101, size=10),
    'Grade': ''  # Initially empty
}
df = pd.DataFrame(data)

print("Initial DataFrame (Generated):\n", df)

# Function to assign grade based on score
def assign_grade(score):
    if 90 <= score <= 100:
        return 'A'
    elif 80 <= score <= 89:
        return 'B'
    elif 70 <= score <= 79:
        return 'C'
    elif 60 <= score <= 69:
        return 'D'
    else:
        return 'F'

# Apply grade assignment to the DataFrame
df['Grade'] = df['Score'].apply(assign_grade)

print("\nDataFrame with Assigned Grades:\n", df)

# Display the DataFrame sorted by 'Score' in descending order
sorted_df = df.sort_values(by='Score', ascending=False)
print("\nDataFrame Sorted by Score (Descending):\n", sorted_df)

# Calculate and print the average score for each subject
avg_scores = df.groupby('Subject')['Score'].mean().round(2)
print("\nAverage Score per Subject:\n", avg_scores)

# Function to filter students with grades A or B
def pandas_filter_pass(dataframe):
    """
    Filters the provided DataFrame to return only records with grades 'A' or 'B'.
    
    Parameters:
        dataframe (pd.DataFrame): Input DataFrame containing student records.
        
    Returns:
        pd.DataFrame: Filtered DataFrame with grades A or B.
    """
    return dataframe[dataframe['Grade'].isin(['A', 'B'])].reset_index(drop=True)

# Get DataFrame with students who passed with A or B
passed_students = pandas_filter_pass(df)
print("\nStudents with Grades A or B:\n", passed_students)
