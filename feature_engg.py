import streamlit as st
from annotated_text import annotation
import annotated_text 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from streamlit_extras.stylable_container import stylable_container
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import time
from category_encoders import BinaryEncoder, TargetEncoder
from sklearn.cluster import DBSCAN
from scipy import stats

def main():
              list_tabs = ["Outlier Detection", "Encoding Techniques", "Feature Scaling"]
              tab1, tab2, tab3 = st.tabs(list_tabs)
              with tab1:
                st.markdown("<h1 style='color: #1F77B4; font-size: 2.5em;'>Unmasking Outliers: The Hidden Stories in Your Data</h1>", unsafe_allow_html=True)
                st.write("""**An outlier** is a data point that significantly differs from the majority of observations in a dataset. It can be much higher or lower than the other values and may indicate variability in the data or an experimental error.""")

                st.write(
                        """
                        Imagine you are in a classroom where most students are between 4.5 and 5.5 feet tall. 
                        But then, one student is **8** feet tall, and another is **2** feet tall. These students are 
                        much different in height compared to the rest of the class. 
                        In statistics, we call these unusual or extreme values **outliers**.
                        """
                    )
                data = np.random.normal(5, 0.5, 28)  # Heights clustered around 5 feet
                data = np.append(data, [8, 2]) # Adding Outliers

                # Display data
                st.write("Here’s a dataset of heights:")
                st.dataframe(pd.DataFrame(data, columns=["Height"]).T, hide_index = True)

                # Plot the data
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.boxplot(data, vert=False, patch_artist=True)
                ax.set_title("Boxplot Highlighting Outliers")
                ax.set_xlabel("Height")
                st.pyplot(fig)

                # Explanation
                st.write(
                    """
                    The boxplot above shows most of the heights clustered together in a range. The dots 
                    outside the "box" are the **outliers**. These are the values that are far from the rest.
                    """
                )
                # Interaction: Highlight or exclude outliers
                show_outliers = st.checkbox("Highlight Outliers")
                if show_outliers:
                    mean = np.mean(data)
                    std_dev = np.std(data)
                    outliers = data[(data > mean + 3 * std_dev) | (data < mean - 3 * std_dev)]
                    st.write("Outliers detected in the data:", outliers)

                st.markdown("""
                    ### Characteristics of Outliers

                    1. **Extreme values**: 
                    Outliers lie far away from the central tendency (mean, median, mode) of the data.
                    
                    2. **Influence on analysis**: 
                    Outliers can skew statistical calculations, such as the mean and standard deviation, potentially leading to misleading results.
                    
                    3. **Impact on models**: 
                    In predictive modeling, outliers may disproportionately affect regression lines, clustering, or classification boundaries.
                    """)
                
                st.markdown("""
                    ### Outlier Impact on Algorithms:
                """)

                data = {
                    "Algorithm": [
                        "Linear Regression", 
                        "Logistic Regression", 
                        "K-Means Clustering", 
                        "PCA", 
                        "Decision Trees", 
                        "Random Forests", 
                        "DBSCAN"
                    ],
                    "Outlier Impact": [
                        "High", "High", "High", "High", 
                        "Low", "Low", "Low"
                    ],
                    "Reason": [
                        "Outliers greatly affect the line of best fit.",
                        "Outliers can distort the calculations.",
                        "Outliers pull the cluster centers away.",
                        "Outliers dominate the variation in the data.",
                        "Outliers don’t affect how the splits are made.",
                        "Combines many trees, reducing outlier effects.",
                        "Marks outliers as noise instead of clustering them."
                    ]
                }

                # Convert to DataFrame
                df = pd.DataFrame(data)

                st.table(df)

                st.markdown("""
                    ### Outlier Treatment Techniques:
                """)

                st.write("""
                    Outliers can negatively impact analysis or models, but they can also provide valuable insights. 
                    Below are common techniques for handling outliers and when to use them.
                    """)
                
                # Z-Score Explanation
                st.subheader("Z-Score: Detecting Outliers in Normally Distributed Data")
                st.write("**Formula:** Z = (X - μ) / σ")
                st.write("Where:")
                st.write("- X: Data point")
                st.write("- μ: Mean of the dataset")
                st.write("- σ: Standard deviation of the dataset")
                st.write("""
                - **Best For**: Data that follows a normal distribution.
                - **Threshold**: Commonly, \( |Z| > 3 \).
                """)

                # Generate sample data for Z-Score visualization
                np.random.seed(42)
                data = np.random.normal(0, 1, 1000)
                outliers = np.array([5, -5])  # Simulated outliers
                data_with_outliers = np.concatenate([data, outliers])

                # Plot Z-Score Visualization
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(data_with_outliers, bins=30, alpha=0.7, color='blue', label='Data')
                ax.axvline(x=3, color='red', linestyle='--', label='Z = 3 (Threshold)')
                ax.axvline(x=-3, color='red', linestyle='--')
                ax.set_title("Z-Score Outlier Detection")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.legend()
                st.pyplot(fig)

                # IQR Explanation
                st.subheader("IQR: Detecting Outliers in Skewed Data")
                st.write("""
                - **Definition**:
                \[
                IQR = Q3 - Q1
                \]
                Outliers are identified as:
                - Below \( Q1 - 1.5 \ttimes IQR \)
                - Above \( Q3 + 1.5 \ttimes IQR \)
                - **Best For**: Data that is skewed or not normally distributed.
                """)

                # Generate sample data for IQR visualization
                np.random.seed(42)
                skewed_data = np.random.exponential(scale=1, size=1000)

                # Calculate IQR
                Q1 = np.percentile(skewed_data, 25)
                Q3 = np.percentile(skewed_data, 75)
                Q2 = np.percentile(skewed_data, 50)  # Median
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Plot IQR Visualization with Q1, Q2, Q3 markers
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.boxplot(skewed_data, vert=False, patch_artist=True, boxprops=dict(facecolor="blue", color="blue"))
                ax.axvline(x=lower_bound, color='red', linestyle='--', label='Lower Bound')
                ax.axvline(x=upper_bound, color='red', linestyle='--', label='Upper Bound')
                ax.axvline(x=Q1, color='green', linestyle='-', label='Q1 (25th Percentile)')
                ax.axvline(x=Q2, color='orange', linestyle='-', label='Q2 (50th Percentile - Median)')
                ax.axvline(x=Q3, color='purple', linestyle='-', label='Q3 (75th Percentile)')

                # Adding Labels
                ax.text(Q1, 1.05, 'Q1', color='green', ha='center')
                ax.text(Q2, 1.05, 'Q2', color='orange', ha='center')
                ax.text(Q3, 1.05, 'Q3', color='purple', ha='center')

                # Customize plot appearance
                ax.set_title("IQR Outlier Detection with Q1, Q2, Q3")
                ax.set_xlabel("Value")
                ax.legend()
                st.pyplot(fig)

                
                # Techniques Data
                techniques = [
                    {
                        "Technique": "Removing Outliers",
                        "Description": "Exclude outliers from the dataset.",
                        "When to Use": "For errors, noise, or invalid data.",
                        "Methods": "Z-Score (|Z| > 3), IQR (Q1 - 1.5 \ttimes IQR or Q3 + 1.5 \ttimes IQR)."
                    },
                    {
                        "Technique": "Capping or Clipping",
                        "Description": "Limit extreme values to fixed boundaries.",
                        "When to Use": "When outliers are valid but need scaling.",
                        "Methods": "Replace extremes with boundaries (e.g., 5th/95th percentiles)."
                    },
                    {
                        "Technique": "Transforming Data",
                        "Description": "Reduce the impact of extreme values through transformations.",
                        "When to Use": "When valid outliers skew the data.",
                        "Methods": "Log, square root, or Box-Cox transformations."
                    },
                    {
                        "Technique": "Imputation",
                        "Description": "Replace outliers with representative values.",
                        "When to Use": "When removal or capping isn't suitable.",
                        "Methods": "Use median, mean, or mode to replace outliers."
                    },
                    {
                        "Technique": "Treating Outliers as a Separate Class",
                        "Description": "Analyze outliers separately as anomalies.",
                        "When to Use": "In anomaly or fraud detection.",
                        "Methods": "Isolation Forest, DBSCAN, one-class SVM."
                    },
                    {
                        "Technique": "Binning",
                        "Description": "Group data into bins to minimize the effect of outliers.",
                        "When to Use": "For categorical analysis or when precision isn't critical.",
                        "Methods": "Create bins for data ranges and assign outliers to boundary bins."
                    },
                    {
                        "Technique": "Ignoring Outliers",
                        "Description": "Leave outliers as they are.",
                        "When to Use": "When outliers are meaningful or rare phenomena.",
                        "Methods": "Analyze separately to gain insights."
                    }
                ]

                # Display techniques in Streamlit
                for technique in techniques:
                    st.subheader(technique["Technique"])
                    st.write(f"**Description:** {technique['Description']}")
                    st.write(f"**When to Use:** {technique['When to Use']}")
                    st.write(f"**Methods:** {technique['Methods']}")
                    st.markdown("---")  # Divider between techniques


                # Generate base data
                np.random.seed(42)
                data = np.random.normal(0, 1, 1000)  # Normal distribution
                outliers = np.array([5, -5, 6, -6])  # Simulated outliers
                data_with_outliers = np.concatenate([data, outliers])

                # Create a function for plotting
                def plot_data(data, title="Data Visualization", xlabel="Value", ylabel="Frequency"):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(data, kde=True, ax=ax, color='skyblue')
                    ax.set_title(title)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    st.pyplot(fig)

                # Show original data with outliers
                st.subheader("Try it yourself")
                plot_data(data_with_outliers, title="Original Data with Outliers")

                # Technique selection directly on the screen
                st.subheader("Choose an Outlier Treatment Technique")
                treatment_choice = st.selectbox(
                    "Choose an outlier treatment technique:",
                    ["None", "Remove Outliers", "Cap Outliers", "Transform Data", "Impute Outliers", "Treat Outliers as Class", "Binning", "Ignore Outliers"]
                )

                # Technique 1: Remove Outliers
                if treatment_choice == "Remove Outliers":
                    z_scores = stats.zscore(data_with_outliers)
                    cleaned_data = data_with_outliers[np.abs(z_scores) < 3]
                    st.subheader("Removing Outliers (Z-Score > 3)")
                    plot_data(cleaned_data, title="Data after Removing Outliers")

                # Technique 2: Cap Outliers
                elif treatment_choice == "Cap Outliers":
                    lower_cap = np.percentile(data_with_outliers, 5)
                    upper_cap = np.percentile(data_with_outliers, 95)
                    capped_data = np.clip(data_with_outliers, lower_cap, upper_cap)
                    st.subheader("Capping or Clipping Outliers")
                    plot_data(capped_data, title="Data after Capping Outliers")

                # Technique 3: Transform Data (Log, Square Root)
                elif treatment_choice == "Transform Data":
                    transform_type = st.sidebar.selectbox("Choose transformation:", ["Log", "Square Root"])
                    if transform_type == "Log":
                        transformed_data = np.log1p(data_with_outliers)
                        st.subheader("Log Transformation")
                    else:
                        transformed_data = np.sqrt(data_with_outliers)
                        st.subheader("Square Root Transformation")
                    plot_data(transformed_data, title=f"Data after {transform_type} Transformation")

                # Technique 4: Impute Outliers
                elif treatment_choice == "Impute Outliers":
                    median_value = np.median(data_with_outliers)
                    imputed_data = np.where(np.abs(stats.zscore(data_with_outliers)) > 3, median_value, data_with_outliers)
                    st.subheader("Imputing Outliers with Median Value")
                    plot_data(imputed_data, title="Data after Imputing Outliers")

                # Technique 6: Treat Outliers as a Separate Class (Using DBSCAN)
                elif treatment_choice == "Treat Outliers as Class":
                    db = DBSCAN(eps=0.5, min_samples=5).fit(data_with_outliers.reshape(-1, 1))
                    labels = db.labels_
                    outlier_data = data_with_outliers[labels == -1]
                    st.subheader("Treat Outliers as Separate Class")
                    plot_data(outlier_data, title="Detected Outliers by DBSCAN")

                # Technique 7: Binning
                elif treatment_choice == "Binning":
                    bins = np.linspace(min(data_with_outliers), max(data_with_outliers), 10)
                    binned_data = np.digitize(data_with_outliers, bins)
                    st.subheader("Binning Outliers")
                    plot_data(binned_data, title="Data after Binning")

                # Technique 8: Ignore Outliers
                elif treatment_choice == "Ignore Outliers":
                    st.subheader("Ignoring Outliers")
                    plot_data(data_with_outliers, title="Data with Ignored Outliers")

                # Showing summary of chosen technique
                st.subheader(f"Chosen Technique: {treatment_choice}")
                st.write("""
                - **Removing Outliers**: Outliers are removed if their Z-score is greater than 3.
                - **Capping or Clipping**: Outliers are replaced with the lower or upper 5th/95th percentiles.
                - **Transforming Data**: Applying log or square root transformations reduces the impact of outliers.
                - **Impute Outliers**: Outliers are replaced with the median value of the dataset.
                - **Treat Outliers as a Separate Class**: Outliers are detected and treated as a separate class using DBSCAN.
                - **Binning**: Data is divided into bins, reducing the impact of extreme values.
                - **Ignore Outliers**: Outliers are left untouched in the dataset.
                """)

                # Conclusion Section
                st.subheader("Conclusion")

                st.write("""
                Outliers are a natural part of any dataset and can arise due to various reasons, such as data entry errors, rare events, or unique characteristics. While outliers may provide valuable insights into extreme events or anomalies, they can also distort statistical analysis and impact the performance of machine learning models.

                Choosing the right technique for handling outliers depends on the type of data, the model you're using, and the goal of your analysis. Whether it's removing outliers, capping their values, transforming the data, or using robust methods that reduce their impact, each approach has its benefits and trade-offs.

                It's important to remember that not all outliers should be treated the same way. In some cases, they may contain important information, while in others, they may simply be noise that needs to be addressed. By carefully considering the impact of outliers on your analysis and model, you can ensure more accurate and reliable results.

                As always, experimenting with different techniques and visualizing the effects can help you make informed decisions. Ultimately, the choice of outlier treatment should align with the objectives of your project, ensuring that the insights derived are as accurate and meaningful as possible.
                """)

                st.divider()
                ## Practice Questions on Outliers
                st.subheader("Practice Questions:")

                with st.expander("Outlier Detection Questions:"):
                    questions = [
                        {
                            "question": "What is the purpose of detecting outliers in a dataset?",
                            "options": [
                                "To identify errors or anomalies in data",
                                "To increase the variability of the dataset",
                                "To find the highest values in the data",
                                "To adjust the data distribution"
                            ],
                            "correct_answer": "To identify errors or anomalies in data"
                        },
                        {
                            "question": "Which of the following methods is commonly used to detect outliers?",
                            "options": [
                                "Z-score",
                                "K-means clustering",
                                "Principal Component Analysis (PCA)",
                                "Hierarchical clustering"
                            ],
                            "correct_answer": "Z-score"
                        },
                        {
                            "question": "If a Z-score of a data point is greater than 3 or less than -3, what does this indicate?",
                            "options": [
                                "The data point is likely an outlier",
                                "The data point is part of the normal distribution",
                                "The data point is a central value",
                                "The data point needs further investigation"
                            ],
                            "correct_answer": "The data point is likely an outlier"
                        },
                        {
                            "question": "What is the IQR method for outlier detection based on?",
                            "options": [
                                "The difference between the 25th percentile (Q1) and the 75th percentile (Q3)",
                                "The difference between the mean and standard deviation",
                                "The minimum and maximum values in the dataset",
                                "The median of the dataset"
                            ],
                            "correct_answer": "The difference between the 25th percentile (Q1) and the 75th percentile (Q3)"
                        },
                        {
                            "question": "Which of the following is the most appropriate method for dealing with outliers in a dataset?",
                            "options": [
                                "Removing the outliers",
                                "Replacing outliers with the mean",
                                "Transforming the data",
                                "Imputing missing values"
                            ],
                            "correct_answer": "Transforming the data"
                        },
                        {
                            "question": "When applying the IQR method, which of the following is considered an outlier?",
                            "options": [
                                "Any data point outside the range of (Q1 - 1.5*IQR, Q3 + 1.5*IQR)",
                                "Any data point above the mean",
                                "Any data point above Q3",
                                "Any data point within 1 standard deviation"
                            ],
                            "correct_answer": "Any data point outside the range of (Q1 - 1.5*IQR, Q3 + 1.5*IQR)"
                        },
                        {
                            "question": "How can transforming the data help in handling outliers?",
                            "options": [
                                "It makes the data more normally distributed",
                                "It increases the number of outliers",
                                "It reduces the data variability",
                                "It removes all outliers from the dataset"
                            ],
                            "correct_answer": "It makes the data more normally distributed"
                        },
                        {
                            "question": "Which of the following is a common transformation to handle outliers?",
                            "options": [
                                "Logarithmic transformation",
                                "Exponential transformation",
                                "Square root transformation",
                                "All of the above"
                            ],
                            "correct_answer": "Logarithmic transformation"
                        },
                        {
                            "question": "Which of the following techniques is NOT typically used for detecting outliers?",
                            "options": [
                                "Boxplot",
                                "Scatter plot",
                                "Histogram",
                                "K-means clustering"
                            ],
                            "correct_answer": "K-means clustering"
                        },
                        {
                            "question": "What action should you take if you identify outliers that cannot be removed or transformed?",
                            "options": [
                                "Investigate if they represent a specific group or event",
                                "Ignore them and continue with the analysis",
                                "Treat them as missing values",
                                "Automatically remove them from the dataset"
                            ],
                            "correct_answer": "Investigate if they represent a specific group or event"
                        }
                    ]

                    # Function to display the quiz
                    def display_quiz():
                        score = 0
                        responses = []
                        incorrect_answers = []

                        # Loop through each question and display it
                        for i, question in enumerate(questions):
                            st.caption(f"Question {i+1}:")
                            
                            # The key remains, but there's no locking of answers after selection
                            answer = st.radio(question["question"], options=question["options"], key=f"qqq{i}")

                            # Check the answer and store whether it's correct or not
                            if answer:
                                if answer == question["correct_answer"]:
                                    responses.append(True)
                                else:
                                    responses.append(False)
                                    incorrect_answers.append((i + 1, question["question"], answer, question["correct_answer"]))

                        # Submit button to calculate score
                        if st.button("Submit Test", key="qqq"):
                            score = sum(responses)
                            st.write(f"Your score is: {score}/10")
                            st.session_state["test_done"] = True

                            # Display incorrect answers with correct answers
                            if incorrect_answers:
                                st.write("You got the following questions wrong:")
                                for question_number, question_text, user_answer, correct_answer in incorrect_answers:
                                    st.divider()
                                    st.write(f"Question {question_number}: {question_text}")
                                    st.markdown(f"<span style='color:red'>Your Answer: {user_answer}</span>", unsafe_allow_html=True)
                                    st.markdown(f"<span style='color:green'>Correct Answer: {correct_answer}</span>", unsafe_allow_html=True)
                            else:
                                st.write("Great job! You answered all questions correctly.")

                    # Initialize session state for the first time
                    if "test_done" not in st.session_state:
                        st.session_state["test_done"] = False

                    display_quiz()
              with tab2:
                st.markdown("<h1 style='color: #1F77B4; font-size: 2.5em;'>Welcome to the Encoding Techniques Explorer!</h1>", unsafe_allow_html=True)

                # st.title("Welcome to the Encoding Techniques Explorer!")
                st.subheader("Learn how to transform data into a form that models can understand and use.")

                st.write("In data science, encoding is the process of converting text, labels, or categorical data into numerical forms that computers can understand. This is crucial for using non-numerical data in data analysis and machine learning models.")
                st.write("Imagine teaching a robot to understand different types of fruit. The robot doesn’t understand ‘Apple’ or ‘Banana,’ but if we label them as numbers, like ‘Apple = 1’ and ‘Banana = 2,’ it can process that information. Encoding is like assigning each fruit a unique number or pattern so the robot can learn from it!")
                st.write("Machine learning models need numbers, not words or labels. Encoding makes it possible for models to recognize patterns, make predictions, and analyze data by converting categories into a language they can understand.")

                st.divider()

                st.subheader("Common Encoding Techniques:🛠️")

                encoding_choice = st.selectbox(
                    "Select an encoding technique to learn more about:",
                    ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding", "Frequency Encoding", "Binary Encoding", "Target Encoding"]
                )

                # Information based on selected encoding technique
                if encoding_choice == "Label Encoding":
                    st.markdown("### 1. Label Encoding 🔢")
                    st.write("Label Encoding assigns each unique category a distinct integer.")
                    st.write("**Note:** This technique is commonly used for encoding **output** features (target labels) in classification tasks.")
                    
                    # Example
                    st.write("Example: if we have three fruits — Apple, Banana, and Cherry — Label Encoding would assign")
                    st.write("Fruits - Apple = 0, Banana = 1, Cherry = 2")

                    st.caption("To apply the Label Encoding on our data we use sklearn library:")
                    st.code("""
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            df[column] = le.fit_transform(df[column])
                            """)

                elif encoding_choice == "One-Hot Encoding":
                    st.markdown("### 2. One-Hot Encoding 🟩")
                    st.write("One-Hot Encoding creates a separate binary column for each category, using 1 or 0 to indicate the presence of that category.")
                    
                    # Enhanced Example in Tabular Format
                    st.write("Example:")
                    data = {
                        "Fruit": ["Apple", "Banana", "Cherry"],
                        "Apple": [1, 0, 0],
                        "Banana": [0, 1, 0],
                        "Cherry": [0, 0, 1]
                    }
                    df = pd.DataFrame(data)
                    st.dataframe(df, hide_index = True)

                    st.markdown("#### ⚠️ Dummy Variable Trap Explained")
                    st.write(
                        "When we apply One-Hot Encoding (OHE), the sum of each row will be 1. "
                        "This means there's a relationship between the columns, which can lead to multicollinearity—a problem where input columns are highly correlated. "
                        "To avoid this, we drop the first column, breaking the dependency and ensuring no relationship exists among the remaining columns. "
                        "The columns created from One-Hot Encoding are called 'dummy variables,' so dropping one helps prevent this Dummy Variable Trap."
                    )

                    # Example after removing the Dummy Variable Trap
                    st.write("With the 'Apple' column dropped, the data now looks like this:")
                    data_no_trap = {
                        "Fruit": ["Apple", "Banana", "Cherry"],
                        "Banana": [0, 1, 0],
                        "Cherry": [0, 0, 1]
                        # 'Apple' column is removed
                    }
                    st.dataframe(pd.DataFrame(data_no_trap), hide_index = True)

                    # Additional note for many categories
                    st.markdown("#### 📝 Handling Many Categories")
                    st.write(
                        "If a column has many categories, creating separate columns for each one can make the dataset very large and complex. "
                        "In such cases, we often keep only the most important categories and group all other, less frequent ones into a single category called 'Others'."
                    )

                    st.caption("To apply the One-Hot Encoding on our data we use sklearn library:")
                    st.code("""
                            from sklearn.preprocessing import OneHotEncoder
                            ohe = OneHotEncoder()
                            df[column] = ohe.fit_transform(df[column])
                            """)
                    st.caption("or we can use pandas get_dummies function instead")
                    st.code("""
                            ohe = pd.get_dummies(df[column], drop_first=True)
                            df = df.join(ohe)
                            df.drop(column, axis=1, inplace=True)
                            """)


                elif encoding_choice == "Ordinal Encoding":
                    st.markdown("### 3. Ordinal Encoding 📏")
                    st.write("Ordinal Encoding is used for categories with a meaningful order or ranking. It assigns an integer value to each category based on its order.")
                    
                    # Example with Spice Levels
                    st.write("Example:")
                    st.write("Spice Levels - Mild = 1, Medium = 2, Hot = 3")

                    st.caption("To apply the Ordinal Encoding on our data we use sklearn library:")
                    st.code("""
                            from sklearn.preprocessing import OrdinalEncoder
                            oe = OrdinalEncoder(categories=[order])
                            df[column] = oe.fit_transform(df[column])
                            """)

                elif encoding_choice == "Frequency Encoding":
                    st.markdown("### 4. Frequency Encoding 📊")
                    st.write("Frequency Encoding assigns each category a value based on its frequency in the dataset. This can be helpful when the frequency of categories is meaningful for the model.")
                    
                    # Example
                    st.write("Example:")
                    data = {'Animal': ["Dog", "Cat", "Bird", "Dog", "Dog", "Cat"]}
                    df0 = pd.DataFrame(data)
                    st.dataframe(df0.T)
                    data = {
                        "Animal": ["Dog", "Cat", "Bird"],
                        "Frequency": [3, 2, 1]
                    }
                    df = pd.DataFrame(data)
                    st.dataframe(df, hide_index = True)

                    st.caption("To apply the Frequency Encoding:")
                    st.code("""
                            freq_map = df[column].value_counts().to_dict()
                            df[column] = df[column].map(freq_map)
                            """)
                    
                elif encoding_choice == "Binary Encoding":
                    st.markdown("### 4. Binary Encoding 📊")
                    st.write(
                        "Binary Encoding is useful for categorical variables with a high cardinality (many unique categories). It combines the features of both one-hot encoding and integer encoding. In Binary Encoding, each category is first assigned an integer value, and then that integer is converted into its binary form. This technique is more efficient in terms of memory usage, as it uses fewer columns than one-hot encoding while avoiding the problem of ordering in label encoding."
                    )
                    
                    # Example
                    st.write("Example:")
                    data = {'Animal': ["Dog", "Cat", "Bird", "Dog", "Dog", "Cat"]}
                    df0 = pd.DataFrame(data)
                    st.dataframe(df0.T)

                    # Apply Binary Encoding to the 'Animal' column
                    encoder = BinaryEncoder(cols=['Animal'])
                    df_encoded = encoder.fit_transform(df0)
                    st.write("Binary Encoding applied:")
                    st.dataframe(df_encoded, hide_index=True)

                    st.caption("To apply Binary Encoding, you can use the `category_encoders` library:")

                    st.code("""
                        from category_encoders import BinaryEncoder
                        encoder = BinaryEncoder(cols=['Animal'])
                        df_encoded = encoder.fit_transform(df0)
                        """)
                    
                    st.write(
                        "In the transformed DataFrame, each category in the 'Animal' column has been encoded into binary digits, which are represented as separate columns. This reduces the number of features compared to one-hot encoding, especially for categorical variables with many categories."
                    )

                elif encoding_choice == "Target Encoding":
                    st.markdown("### 5. Target Encoding 📊")
                    st.write(
                        "Target Encoding (also known as Mean Encoding) replaces each category with the mean of the target variable for that category. This technique is particularly useful when the categories have different distributions of the target variable. It is often used in situations where one-hot encoding might cause the model to be overwhelmed by too many sparse columns."
                    )
                    
                    # Example
                    st.write("Example:")
                    data = {'Animal': ["Dog", "Cat", "Bird", "Dog", "Dog", "Cat"],
                            'Price': [100, 150, 50, 120, 110, 160]}
                    df0 = pd.DataFrame(data)
                    st.dataframe(df0.T)
                    
                    st.write("In this example, we will target encode the 'Animal' column using the 'Price' column as the target.")

                    # Initialize and apply Target Encoding
                    encoder = TargetEncoder(cols=['Animal'])
                    df_encoded = encoder.fit_transform(df0['Animal'], df0['Price'])
                    df0['Animal_Encoded'] = df_encoded
                    st.write("Target Encoding applied:")
                    st.dataframe(df0, hide_index=True)

                    st.caption("To apply Target Encoding, you can use the `category_encoders` library:")

                    st.code("""
                        from category_encoders import TargetEncoder
                        encoder = TargetEncoder(cols=['Animal'])
                        df_encoded = encoder.fit_transform(df0['Animal'], df0['Price'])
                        df0['Animal_Encoded'] = df_encoded
                        """)

                    st.write(
                        "In the transformed DataFrame, the 'Animal' column is replaced by the mean of the 'Price' for each category (Dog, Cat, Bird). This allows the model to learn the relationship between the categorical variable and the target variable, which can improve performance on some machine learning models."
                    )



                # Prompt to explore each encoding technique interactively
                st.write("**Choose another encoding technique from the dropdown to learn more!")

                st.divider()

                st.subheader("Practical Implementation:")

                def label_encode(df, column):
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])
                    return df

                # Function to apply One-Hot Encoding
                # def one_hot_encode(df, column):
                #     # ohe = pd.get_dummies(df[column], drop_first=True)
                #     # df = df.join(ohe)
                #     # df.drop(column, axis=1, inplace=True)
                #     # return df
                #   ohe = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid dummy variable trap
                #   encoded_array = ohe.fit_transform(df[[column]])
                #   encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out([column]))
                #   encoded_df.index = df.index  # Align the index with the original DataFrame
              
                #   # Join the new columns to the original DataFrame and drop the original column
                #   df = df.join(encoded_df).drop(column, axis=1)
                  
                #   return df
                def one_hot_encode(df, column):
                  # Adjust for scikit-learn versions 1.2 and above
                  ohe = OneHotEncoder(sparse_output=False, drop='first')  # Updated parameter name
                  encoded_array = ohe.fit_transform(df[[column]])  # Pass as 2D array
              
                  # Create a DataFrame with the one-hot encoded columns
                  encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out([column]))
                  encoded_df.index = df.index  # Align the index with the original DataFrame
              
                  # Join the new columns to the original DataFrame and drop the original column
                  df = df.join(encoded_df).drop(column, axis=1)
                  
                  return df

                # Function to apply Ordinal Encoding
                def ordinal_encode(df, column, order):
                    try:
                        ordinal_encoder = OrdinalEncoder(categories=[order])
                        df[column] = ordinal_encoder.fit_transform(df[[column]])
                        return df
                    except:
                        return None
                
                # Function to apply Frequency Encoding
                def frequency_encode(df, column):
                    freq_map = df[column].value_counts().to_dict()
                    df[column] = df[column].map(freq_map)
                    return df
                    
                def binary_encode(df, column):
                    try:
                        binary_encoder = BinaryEncoder(cols=[column])
                        df_encoded = binary_encoder.fit_transform(df)
                        return df_encoded
                    except Exception as e:
                        st.error(f"Error during binary encoding: {e}")
                        return None
                    
                # def target_encode(df, column, target_column):
                #     try:
                #         target_encoder = TargetEncoder(cols=[column])
                        
                #         # Fit and transform the data (the entire DataFrame, not just the column)
                #         df[column] = target_encoder.fit_transform(df[column], df[target_column])  # Proper usage

                #         return df
                #     except Exception as e:
                #         st.error(f"Error during target encoding: {e}")
                #         return None

                # File uploader widget
                uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

                # Sample data if no file uploaded
                if uploaded_file is None:
                    st.write("No file uploaded, using a sample dataset.")
                    data = {
                        'Fruit': ['Apple', 'Banana', 'Cherry', 'Apple', 'Banana', 'Cherry', 'Banana', 'Apple'],
                        'SpiceLevel': ['Mild', 'Medium', 'Hot', 'Mild', 'Medium', 'Hot', 'Mild', 'Medium'],
                    }
                    df = pd.DataFrame(data)
                else:
                    # Load user-uploaded dataset
                    df = pd.read_csv(uploaded_file)

                st.write("Dataset Preview:")
                st.dataframe(df, hide_index = True)

                # Select encoding technique
                encoding_option = st.selectbox("Select Encoding Technique", 
                                            ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding", "Frequency Encoding", "Binary Encoding"])

                # Handle encoding options
                if encoding_option == "Label Encoding":
                    column_to_encode = st.selectbox("Select Column to Encode", df.columns)
                    encoded_df = label_encode(df.copy(), column_to_encode)
                    st.write(f"Encoded Data (Label Encoding on {column_to_encode}):")
                    st.dataframe(encoded_df, hide_index=True)

                elif encoding_option == "One-Hot Encoding":
                    column_to_encode = st.selectbox("Select Column to Encode", df.columns)
                    encoded_df = one_hot_encode(df.copy(), column_to_encode)
                    st.write(f"Encoded Data (One-Hot Encoding on {column_to_encode}):")
                    st.dataframe(encoded_df, hide_index=True)

                elif encoding_option == "Ordinal Encoding":
                    column_to_encode = st.selectbox("Select Column to Encode", df.columns)
                    order = st.text_input("Enter the order for encoding (comma separated)", "Mild,Medium,Hot").split(',')
                    order = [x.strip() for x in order]  # Clean up the list
                    encoded_df = ordinal_encode(df.copy(), column_to_encode, order)
                    st.write(f"Encoded Data (Ordinal Encoding on {column_to_encode}):")
                    st.dataframe(encoded_df, hide_index=True)

                elif encoding_option == "Frequency Encoding":
                    column_to_encode = st.selectbox("Select Column to Encode", df.columns)
                    encoded_df = frequency_encode(df.copy(), column_to_encode)
                    st.write(f"Encoded Data (Frequency Encoding on {column_to_encode}):")
                    st.dataframe(encoded_df, hide_index=True)

                elif encoding_option == "Binary Encoding":
                    column_to_encode = st.selectbox("Select Column to Encode", df.columns)
                    encoded_df = binary_encode(df.copy(), column_to_encode)
                    if encoded_df is not None:
                        st.write(f"Encoded Data (Binary Encoding on {column_to_encode}):")
                        st.dataframe(encoded_df, hide_index=True)

                # elif encoding_option == "Target Encoding":
                #     column_to_encode = st.selectbox("Select Column to Encode", st.session_state.df.columns)
                #     target_column = st.selectbox("Select Target Column", st.session_state.df.columns)
                #     encoded_df = target_encode(st.session_state.df.copy(), column_to_encode, target_column)
                #     if encoded_df is not None:
                #         st.write(f"Encoded Data (Target Encoding on {column_to_encode}):")
                #         st.dataframe(encoded_df, hide_index=True)

                st.divider()

                st.subheader("More about Encoding Techniques:")
                with st.expander("Open to see:"):
                # Section for Label Encoding
                    st.markdown("<h5 style='font-weight: bold; text-decoration: underline;'>1. Label Encoding 📊</h5>", unsafe_allow_html=True)
                    st.write("""
                    **Advantages:**
                    - Simple and easy to implement.
                    - Suitable for models that can handle ordinal relationships (e.g., decision trees).
                    - Does not increase dimensionality.

                    **Disadvantages:**
                    - Imposes an ordinal relationship, which may not exist between categories, potentially affecting models that cannot handle ordinal data correctly.

                    **When to Use:**
                    - When the categorical variable has an inherent ordinal relationship (e.g., Low, Medium, High).
                    - When using algorithms that can handle ordinal features like decision trees or random forests.
                    """)

                    # Section for One-Hot Encoding
                    st.markdown("<h5 style='font-weight: bold; text-decoration: underline;'>2. One-Hot Encoding  📊</h5>", unsafe_allow_html=True)
                    st.write("""
                    **Advantages:**
                    - No assumptions about relationships between categories.
                    - Works well with algorithms that assume all categories are independent, like logistic regression or Naive Bayes.

                    **Disadvantages:**
                    - Can cause a large number of features when the categorical variable has many unique categories (high cardinality).
                    - Increases the dimensionality of the dataset.

                    **When to Use:**
                    - When there is no ordinal relationship between categories.
                    - For machine learning models that do not handle categorical data directly (e.g., linear models).
                    """)

                    # Section for Ordinal Encoding
                    st.markdown("<h5 style='font-weight: bold; text-decoration: underline;'>3. Ordinal Encoding  📊</h5>", unsafe_allow_html=True)
                    st.write("""
                    **Advantages:**
                    - Suitable for ordinal data where categories have a meaningful order.
                    - Does not increase dimensionality.

                    **Disadvantages:**
                    - Assumes that the categories have a meaningful order, which may not always be the case.
                    - Not suitable for nominal (unordered) categories.

                    **When to Use:**
                    - When the categorical variable has an inherent ordering (e.g., Rating scale: Poor, Average, Good).
                    - Best suited for models that can understand ordinal relationships.
                    """)

                    # Section for Frequency Encoding
                    st.markdown("<h5 style='font-weight: bold; text-decoration: underline;'>4. Frequency Encoding  📊</h5>", unsafe_allow_html=True)
                    st.write("""
                    **Advantages:**
                    - Simple and fast to apply.
                    - Useful when the frequency of categories has an impact on the prediction.

                    **Disadvantages:**
                    - May not capture meaningful relationships between the categorical variable and the target variable.
                    - Can lead to issues if categories have very similar frequencies.

                    **When to Use:**
                    - When the frequency of categories is meaningful or should influence the model.
                    - For machine learning algorithms that do not natively handle categorical variables.
                    """)

                    # Section for Binary Encoding
                    st.markdown("<h5 style='font-weight: bold; text-decoration: underline;'>5. Binary Encoding  📊</h5>", unsafe_allow_html=True)
                    st.write("""
                    **Advantages:**
                    - Reduces dimensionality compared to One-Hot Encoding for high-cardinality categorical variables.
                    - Can be used for both nominal and ordinal data.

                    **Disadvantages:**
                    - Can lead to the creation of multiple binary columns, which might be difficult to interpret.
                    - May not perform well if there is a strong relationship between the categorical variable and the target variable.

                    **When to Use:**
                    - When working with high-cardinality categorical variables.
                    - Useful when there are too many unique categories for one-hot encoding.
                    """)

                    # Section for Target Encoding
                    st.markdown("<h5 style='font-weight: bold; text-decoration: underline;'>6. Target Encoding  📊</h5>", unsafe_allow_html=True)
                    st.write("""
                    **Advantages:**
                    - Helps capture the relationship between the categorical variable and the target variable.
                    - Useful for high-cardinality categorical features, as it does not increase dimensionality.

                    **Disadvantages:**
                    - Risk of overfitting, especially with small datasets or categories with few occurrences.
                    - Can leak information from the target variable if not handled correctly.

                    **When to Use:**
                    - When there is a relationship between the categorical variable and the target variable.
                    - Especially useful for high-cardinality features in tree-based models or gradient boosting models.
                    - Should be applied carefully to avoid data leakage, typically using cross-validation or smoothing techniques.
                    """)                

                st.divider()
                st.subheader("Practice Questions:")
                with st.expander("Encoding Techniques Questions:"):
                    questions = [
                        {
                            "question": "Which encoding technique should be used when dealing with categorical variables that have no intrinsic order?",
                            "options": [
                                "One-Hot Encoding",
                                "Label Encoding",
                                "Binary Encoding",
                                "Target Encoding"
                            ],
                            "correct_answer": "One-Hot Encoding"
                        },
                        {
                            "question": "What is the main disadvantage of Label Encoding?",
                            "options": [
                                "It creates high-dimensional data",
                                "It assumes an ordinal relationship between categories",
                                "It cannot be applied to numerical data",
                                "It requires a large amount of data to work effectively"
                            ],
                            "correct_answer": "It assumes an ordinal relationship between categories"
                        },
                        {
                            "question": "Which encoding technique is best used when categorical variables have a natural ranking order?",
                            "options": [
                                "One-Hot Encoding",
                                "Label Encoding",
                                "Target Encoding",
                                "Frequency Encoding"
                            ],
                            "correct_answer": "Label Encoding"
                        },
                        {
                            "question": "What is the key difference between One-Hot Encoding and Label Encoding?",
                            "options": [
                                "One-Hot Encoding represents categories as binary vectors, while Label Encoding assigns integers to categories",
                                "One-Hot Encoding is suitable for ordinal variables, while Label Encoding is for nominal variables",
                                "One-Hot Encoding is used for target variables, while Label Encoding is used for feature variables",
                                "Label Encoding cannot handle missing values, whereas One-Hot Encoding can"
                            ],
                            "correct_answer": "One-Hot Encoding represents categories as binary vectors, while Label Encoding assigns integers to categories"
                        },
                        {
                            "question": "When should you avoid using One-Hot Encoding?",
                            "options": [
                                "When the categorical variable has too many unique categories",
                                "When the variable is ordinal",
                                "When the variable has few categories",
                                "When the dataset is very small"
                            ],
                            "correct_answer": "When the categorical variable has too many unique categories"
                        },
                        {
                            "question": "Which encoding technique is most suitable for high-cardinality categorical variables?",
                            "options": [
                                "One-Hot Encoding",
                                "Target Encoding",
                                "Label Encoding",
                                "Frequency Encoding"
                            ],
                            "correct_answer": "Target Encoding"
                        },
                        {
                            "question": "What is the primary advantage of Target Encoding?",
                            "options": [
                                "It creates a binary representation of categories",
                                "It works well for high-cardinality categorical variables",
                                "It requires no additional parameters to be set",
                                "It handles missing values automatically"
                            ],
                            "correct_answer": "It works well for high-cardinality categorical variables"
                        },
                        {
                            "question": "Which encoding method is preferred when the categories in a variable are highly imbalanced?",
                            "options": [
                                "One-Hot Encoding",
                                "Target Encoding",
                                "Label Encoding",
                                "Binary Encoding"
                            ],
                            "correct_answer": "Target Encoding"
                        },
                        {
                            "question": "Which of the following encoding techniques is least likely to cause the curse of dimensionality?",
                            "options": [
                                "One-Hot Encoding",
                                "Label Encoding",
                                "Binary Encoding",
                                "Frequency Encoding"
                            ],
                            "correct_answer": "Binary Encoding"
                        },
                        {
                            "question": "What is the main disadvantage of using Frequency Encoding?",
                            "options": [
                                "It can lead to overfitting when the frequency of categories is highly variable",
                                "It does not preserve the ordinal relationship between categories",
                                "It is not scalable for large datasets",
                                "It creates high-dimensional datasets"
                            ],
                            "correct_answer": "It can lead to overfitting when the frequency of categories is highly variable"
                        }
                    ]

                    # Function to display the quiz
                    def display_quiz():
                        score = 0
                        responses = []
                        incorrect_answers = []

                        # Loop through each question and display it
                        for i, question in enumerate(questions):
                            st.caption(f"Question {i+1}:")
                            
                            # The key remains, but there's no locking of answers after selection
                            answer = st.radio(question["question"], options=question["options"], key=f"qq{i}")

                            # Check the answer and store whether it's correct or not
                            if answer:
                                if answer == question["correct_answer"]:
                                    responses.append(True)
                                else:
                                    responses.append(False)
                                    incorrect_answers.append((i + 1, question["question"], answer, question["correct_answer"]))

                        # Submit button to calculate score
                        if st.button("Submit Test", key = "qq"):
                            score = sum(responses)
                            st.write(f"Your score is: {score}/10")
                            st.session_state["test_done"] = True

                            # Display incorrect answers with correct answers
                            if incorrect_answers:
                                st.write("You got the following questions wrong:")
                                for question_number, question_text, user_answer, correct_answer in incorrect_answers:
                                    st.divider()
                                    st.write(f"Question {question_number}: {question_text}")
                                    st.markdown(f"<span style='color:red'>Your Answer: {user_answer}</span>", unsafe_allow_html=True)
                                    st.markdown(f"<span style='color:green'>Correct Answer: {correct_answer}</span>", unsafe_allow_html=True)
                            else:
                                st.write("Great job! You answered all questions correctly.")

                    # Initialize session state for the first time
                    if "test_done" not in st.session_state:
                        st.session_state["test_done"] = False

                    display_quiz()
              with tab3:
                st.subheader("Feature scaling is a technique to standardize the independent features present in the data in a fixed range")
                st.write("### Types of Feature Scaling:")
                with st.expander("1. Standardization"):
                    st.markdown(
                        """
                        <style>
                        .justified-text {
                            text-align: justify;
                        }
                        </style>
                        <p class="justified-text">
                        <b>Standardization</b> (also known as Z-score Normalization) is a technique that transforms data to have a mean of 0 and a standard deviation of 1. This process adjusts each feature so that it follows a standard normal distribution (or approximately normal), which is useful for many machine learning algorithms that assume normally distributed data, such as linear regression and logistic regression.
                        </p>
                        <p class = "justified-text">Reason: Features in a dataset can have vastly different scales. For example, one feature might range from 0 to 1, while another might range from 1,000 to 100,000. If a machine learning model is trained on this data without standardization, the model may give more importance to features with larger numerical ranges simply due to their scale.</p>
                        <p class = "justified-text">Effect: By standardizing, all features are scaled to the same range, which ensures that each feature contributes equally to the model.</p>
                        """,
                        unsafe_allow_html=True
                    )                        
                    st.write("Formula:")
                    st.latex(r'z = \frac{x - \mu}{\sigma}')
                    st.caption("""where:\n
                    x is the original feature value,\n
    μ is the mean of the feature, and\n
    σ is the standard deviation of the feature.""")
                    st.subheader("Example and Visualization:")
                    df = pd.DataFrame({
                                        'Feature': ['X'],
                                        'Value1': [2],
                                        'value2': [4],
                                        'Value3': [6],
                                        'value4': [8],
                                        'Value5': [10]
                                        })
                    st.table(df)
                    st.caption('STEP 1: Calculate the mean of and standard deviation of X')
                    # st.latex(r"\mu_X = 6, \sigma_X = 2.83")
                    data = np.array([2, 4, 6, 8, 10])
                    mean = int(np.mean(data))
                    std_dev = np.std(data)
                    st.write(f"Mean: {mean}, S.D: {std_dev:.2f}")
                    st.caption('STEP 2: Apply standarization for each value of X')

                    # Calculate and display standardized values with steps
                    for i, value in enumerate(data, start=1):
                        standardized_value = (value - mean) / std_dev
                        st.write(
                            f"For Value {i}: "
                            f"({value} - {mean}) / {std_dev:.2f} ≈ {standardized_value:.2f}"
                        )

                    st.caption('Scaled Data')
                    df2 = pd.DataFrame({
                                        'Feature': ['X'],
                                        'Value1': [-1.41],
                                        'value2': [-0.71],
                                        'Value3': [0],
                                        'value4': [0.71],
                                        'Value5': [1.41]
                                        })
                    st.table(df2)

                    # Mean-centered and standardized data
                    # standardized_data = (data - mean) / std_dev

                    # Plot original, mean-centered, and standardized data
                    # fig, ax = plt.subplots(figsize=(5, 3))  # Smaller figure size

                    # ax.plot(data, label='Original Data', marker='o', linestyle='-', color='b', markersize=4, linewidth=1)
                    # ax.plot(standardized_data, label='Standardized Data', marker='o', linestyle=':', color='g', markersize=4, linewidth=1)

                    # ax.set_xlabel('Index', fontsize=8)
                    # ax.set_ylabel('Value', fontsize=8)
                    # ax.set_title('Standardization of Data', fontsize=10)
                    # ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))  # Position legend outside plot for space efficiency

                    # plt.tight_layout()

                    # st.pyplot(fig)

                    st.caption("Quick Question:")
                    st.write("Why do we apply the standardization technique?")
                    options = [
                        "To reduce overfitting",
                        "To center and scale the data for consistency",
                        "To remove missing values",
                        "To increase feature importance"
                    ]

                    # Display the question with options as radio buttons
                    answer = st.radio("Select the correct answer:", options)

                    # Define the correct answer
                    correct_answer = "To center and scale the data for consistency"

                    # Check the answer when the user clicks the submit button
                    if st.button("Submit Answer"):
                        if answer == correct_answer:
                            st.success("Correct! Standardization centers and scales the data for consistency.")
                        else:
                            st.error("Incorrect. Please try again.")
                    st.caption("To apply the standardization on our data we use sklearn library:")
                    st.code("""
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            scaled = scaler.fit_transform(data)
                            """)
                    @st.experimental_dialog(" ", width="large")
                    def view_resources():
                        data = pd.read_csv('https://raw.githubusercontent.com/Nargis45/Feature_Engineering_Streamlit/main/st.csv')
                        data = data.head()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption('Before Standardization')
                            st.dataframe(data, hide_index = True)
                            st.caption("Original Data Distribution")

                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            ax[0].hist(data['Age'], bins=10, color='skyblue', edgecolor='black')
                            ax[0].set_title('Age (Original)')
                            ax[1].hist(data['Fare'], bins=10, color='lightgreen', edgecolor='black')
                            ax[1].set_title('Fare (Original)')
                            st.pyplot(fig)
                        with col2:
                            sc = StandardScaler()
                            new_data = sc.fit_transform(data)
                            new_data = pd.DataFrame(new_data, columns=data.columns)
                            st.caption('After Standardization')
                            st.dataframe(new_data, hide_index = True)

                            st.caption("Standardized Data Distribution")
                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            ax[0].hist(new_data['Age'], bins=10, color='skyblue', edgecolor='black')
                            ax[0].set_title('Age (Standardized)')
                            ax[1].hist(new_data['Fare'], bins=10, color='lightgreen', edgecolor='black')
                            ax[1].set_title('Fare (Standardized)')
                            st.pyplot(fig)
                        st.write("""
                        After standardization, there is no change in the distribution of data but look at scale of both `Age` and `Fare` features. Both features are transformed to have a mean of 0 and a standard deviation of 1
                        """)

                    with stylable_container(
                        key="view",
                        css_styles="""
                        button {
                        background-color: #FF7F50;
                        color: white;
                    }
                    """
                    ):
                        if st.button('See Practical Example'):
                            view_resources()
                    st.write("""Effect on outliers:\n
                                1. Outliers effect do not get reduced\n
    2. Outliers can skew the mean and standard deviation, affecting the scaling.""")
                    st.write("""When to use:\n
                                Standardization is essential for models that calculate distances between points or are sensitive to the magnitude of feature values\n
    Example: Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Principal Component Analysis (PCA), K-Means Clustering, Neural Networks""")
                ## Normalization
                with st.expander("1. Normalization"):
                    st.markdown(
                        """
                        <style>
                        .justified-text {
                            text-align: justify;
                        }
                        </style>
                        <p class="justified-text">
                        <b>Normalization</b> is a technique often applied as part of data preparation for machine learning. The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values or losing information.
                        </p>
                        <p class="justified-text">Reason: Features in a dataset can have vastly different scales. For example, one feature might range from 0 to 1, while another might range from 1,000 to 100,000. If a machine learning model is trained on this data without normalization, the model may give more importance to features with larger numerical ranges simply due to their scale.</p>
                        <p class="justified-text">Effect: By normalizing, all features are scaled to a fixed range, typically between 0 and 1. This ensures that each feature contributes equally to the model, regardless of its original scale.</p>
                        """,
                        unsafe_allow_html=True
                    )        
                    st.markdown("""<b><u>Types of Normalization:</u></b>
                                    <p>1. Min-Max Scaling</p>
                                    <p>2. Mean Normalization Scaling</p>
                                    <p>3. Max Absolute Scaling</p>
                                    <p>4. Robust Scaling</p>""", unsafe_allow_html=True)   

                    # Display original data
                    st.subheader("Explanation with example:")
                    dff = pd.DataFrame({
                                        'Value1': [10],
                                        'value2': [20],
                                        'Value3': [30],
                                        'value4': [40],
                                        'Value5': [50],
                                        'Value6': [60],
                                        'value7': [70],
                                        'Value8': [80],
                                        'value9': [90],
                                        'Value10': [100]
                                        })
                    st.table(dff)  

                    data = {
                        '0': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                    }
                    df = pd.DataFrame(data)
 

                    def min_max_scaling(df):
                        scaler = MinMaxScaler()
                        scaled = scaler.fit_transform(df)
                        return pd.DataFrame(scaled, columns=df.columns)

                    # Function to apply Mean Normalization Scaling
                    def mean_normalization_scaling(df):
                        min_val = df.min()
                        max_val = df.max()
                        mean_val = df.mean()
                        scaled = (df - mean_val) / (max_val - min_val)
                        return scaled

                    # Function to apply Max Absolute Scaling
                    def max_absolute_scaling(df):
                        scaler = MaxAbsScaler()
                        scaled = scaler.fit_transform(df)
                        return pd.DataFrame(scaled, columns=df.columns)

                    # Function to apply Robust Scaling
                    def robust_scaling(df):
                        scaler = RobustScaler()
                        scaled = scaler.fit_transform(df)
                        return pd.DataFrame(scaled, columns=df.columns)

                    # Min-Max Scaling Explanation
                    st.subheader("1. Min-Max Scaling")
                    st.latex(r"""
                    X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}
                    """)
                    st.write("""
                    where:
                    - \(X\) is the original value, 
                    - \(X_{min}\) and \(X_{max}\) are the minimum and maximum values of the feature.

                    This formula scales all values to the range [0, 1].
                    """)
                    scaled_min_max = min_max_scaling(df)  # Skip the 'Feature' column
                    st.write("Scaled Data (Min-Max Scaling):")
                    scaled_min_max.index = ['Value1', 'value2', 'Value3','value4','Value5','Value6','value7','Value8','value9','Value10']
                    st.write(scaled_min_max.T)

                    st.caption("To apply the Min-Max Scaling on our data we use sklearn library:")
                    st.code("""
                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                            scaled = scaler.fit_transform(df)
                            """)
                    @st.experimental_dialog(" ", width="large")
                    def view_resources():
                        data = pd.read_csv('https://raw.githubusercontent.com/Nargis45/Feature_Engineering_Streamlit/main/st.csv')
                        data = data.head()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption('Before Min-Max Scaling')
                            st.dataframe(data, hide_index = True)
                            st.caption("Original Data Distribution")

                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            ax[0].hist(data['Age'], bins=10, color='skyblue', edgecolor='black')
                            ax[0].set_title('Age (Original)')
                            ax[1].hist(data['Fare'], bins=10, color='lightgreen', edgecolor='black')
                            ax[1].set_title('Fare (Original)')
                            st.pyplot(fig)
                        with col2:
                            sc = MinMaxScaler()
                            new_data = sc.fit_transform(data)
                            new_data = pd.DataFrame(new_data, columns=data.columns)
                            st.caption('After Min-Max Scaling')
                            st.dataframe(new_data, hide_index = True)

                            st.caption("Standardized Data Distribution")
                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            ax[0].hist(new_data['Age'], bins=10, color='skyblue', edgecolor='black')
                            ax[0].set_title('Age (Standardized)')
                            ax[1].hist(new_data['Fare'], bins=10, color='lightgreen', edgecolor='black')
                            ax[1].set_title('Fare (Standardized)')
                            st.pyplot(fig)
                        st.write("""
                        After Min-Max Scaling, there is no change in the distribution of data but look at scale of both `Age` and `Fare` features. 
                        """)

                    with stylable_container(
                        key="view",
                        css_styles="""
                        button {
                        background-color: #FF7F50;
                        color: white;
                    }
                    """
                    ):
                        if st.button('See Practical Example', key = 'Min-Max Scaler'):
                            view_resources()
                    st.write("""Effect on outliers:\n
                                Min-Max Scaling is highly sensitive to outliers. Since it scales data based on the minimum and maximum values, outliers can skew the scaling, compressing the majority of the data into a narrow range.""")
                    st.write("""When to use:\n
                                Use when the data is already within a known, bounded range (e.g., [0, 1]), and when outliers are not a major concern or can be handled separately.""")
                    st.divider()

                    # Mean Normalization Scaling Explanation
                    st.subheader("2. Mean Normalization Scaling")
                    st.latex(r"""
                    X_{scaled} = \frac{X - mean(X)}{X_{max} - X_{min}}
                    """)
                    st.write("""
                    where:
                    - \(X\) is the original value, 
                    - \(mean(X)\) is the mean of the feature,
                    - \(X_{max}\) and \(X_{min}\) are the maximum and minimum values of the feature.

                    This formula scales the data to the range [-1, 1].
                    """)
                    scaled_mean_normalization = mean_normalization_scaling(df)
                    st.write("Scaled Data (Mean Normalization):")
                    scaled_mean_normalization.index = ['Value1', 'value2', 'Value3','value4','Value5','Value6','value7','Value8','value9','Value10']
                    st.write(scaled_mean_normalization.T)

                    st.caption("To apply the Mean Normalization Scaling on our data we use this formula:")
                    st.code("""
                            min_val = df.min()
                            max_val = df.max()
                            mean_val = df.mean()
                            scaled = (df - mean_val) / (max_val - min_val)
                            """)
                    @st.experimental_dialog(" ", width="large")
                    def view_resources():
                        data = pd.read_csv('https://raw.githubusercontent.com/Nargis45/Feature_Engineering_Streamlit/main/st.csv')
                        data = data.head()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption('Before Mean Normalization')
                            st.dataframe(data, hide_index = True)
                            st.caption("Original Data Distribution")

                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            ax[0].hist(data['Age'], bins=10, color='skyblue', edgecolor='black')
                            ax[0].set_title('Age (Original)')
                            ax[1].hist(data['Fare'], bins=10, color='lightgreen', edgecolor='black')
                            ax[1].set_title('Fare (Original)')
                            st.pyplot(fig)
                        with col2:
                            min_val = data.min()
                            max_val = data.max()
                            mean_val = data.mean()
                            new_data = (data - mean_val) / (max_val - min_val)
                            new_data = pd.DataFrame(new_data, columns=data.columns)
                            st.caption('After Mean Normalization')
                            st.dataframe(new_data, hide_index = True)

                            st.caption("Standardized Data Distribution")
                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            ax[0].hist(new_data['Age'], bins=10, color='skyblue', edgecolor='black')
                            ax[0].set_title('Age (Standardized)')
                            ax[1].hist(new_data['Fare'], bins=10, color='lightgreen', edgecolor='black')
                            ax[1].set_title('Fare (Standardized)')
                            st.pyplot(fig)
                        st.write("""
                        After Mean Normalization, there is no change in the distribution of data but look at scale of both `Age` and `Fare` features. 
                        """)

                    with stylable_container(
                        key="view",
                        css_styles="""
                        button {
                        background-color: #FF7F50;
                        color: white;
                    }
                    """
                    ):
                        if st.button('See Practical Example', key = 'Mean Normalization Scaler'):
                            view_resources()
                    st.write("""Effect on outliers:\n
                                Mean Normalization can reduce the impact of outliers compared to Min-Max Scaling because it centers the data around the mean. However, extreme outliers can still influence the scaling to some extent.""")
                    st.write("""When to use:\n
                                Use when the data needs to be centered around 0 (e.g., for algorithms that expect symmetric data) and when you need the data scaled between -1 and 1.""")
                    st.divider()

                    # Max Absolute Scaling Explanation
                    st.subheader("3. Max Absolute Scaling")
                    st.latex(r"""
                    X_{scaled} = \frac{X}{|X_{max}|}
                    """)
                    st.write("""
                    where:
                    - \(X\) is the original value, 
                    - \(|X_{max}|\) is the maximum absolute value of the feature.

                    This formula scales the data by dividing each value by the largest absolute value, ensuring values are within the range [-1, 1].
                    """)
                    scaled_max_absolute = max_absolute_scaling(df)
                    st.write("Scaled Data (Max Absolute Scaling):")
                    scaled_max_absolute.index = ['Value1', 'value2', 'Value3','value4','Value5','Value6','value7','Value8','value9','Value10']
                    st.write(scaled_max_absolute.T)

                    st.caption("To apply the Max Absolute Scaling on our data we use sklearn library:")
                    st.code("""
                            from sklearn.preprocessing import MaxAbsScaler
                            scaler = MaxAbsScaler()
                            scaled = scaler.fit_transform(df)
                            """)

                    @st.experimental_dialog(" ", width="large")
                    def view_resources():
                        data = pd.read_csv('https://raw.githubusercontent.com/Nargis45/Feature_Engineering_Streamlit/main/st.csv')
                        data = data.head()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption('Before Max Absolute Scaling')
                            st.dataframe(data, hide_index = True)
                            st.caption("Original Data Distribution")

                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            ax[0].hist(data['Age'], bins=10, color='skyblue', edgecolor='black')
                            ax[0].set_title('Age (Original)')
                            ax[1].hist(data['Fare'], bins=10, color='lightgreen', edgecolor='black')
                            ax[1].set_title('Fare (Original)')
                            st.pyplot(fig)
                        with col2:
                            sc = MaxAbsScaler()
                            new_data = sc.fit_transform(data)
                            new_data = pd.DataFrame(new_data, columns=data.columns)
                            st.caption('After Max Absolute Scaling')
                            st.dataframe(new_data, hide_index = True)

                            st.caption("Standardized Data Distribution")
                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            ax[0].hist(new_data['Age'], bins=10, color='skyblue', edgecolor='black')
                            ax[0].set_title('Age (Standardized)')
                            ax[1].hist(new_data['Fare'], bins=10, color='lightgreen', edgecolor='black')
                            ax[1].set_title('Fare (Standardized)')
                            st.pyplot(fig)
                        st.write("""
                        After Max Absolute Scaling, there is no change in the distribution of data but look at scale of both `Age` and `Fare` features. 
                        """)

                    with stylable_container(
                        key="view",
                        css_styles="""
                        button {
                        background-color: #FF7F50;
                        color: white;
                    }
                    """
                    ):
                        if st.button('See Practical Example', key = 'Max Absolute Scaler'):
                            view_resources()
                    st.write("""Effect on outliers:\n
                                Max Absolute Scaling is less sensitive to outliers because it divides by the maximum absolute value of the feature, not the range. However, large outliers will still influence the scaling if they are far from the rest of the data.""")
                    st.write("""When to use:\n
                                Use when the data is already centered around zero and when sparsity is important, as it does not change the structure of sparse data.""")
                    st.divider()

                    # Robust Scaling Explanation
                    st.subheader("4. Robust Scaling")
                    st.latex(r"""
                    X_{scaled} = \frac{X - median(X)}{IQR}
                    """)
                    st.write("""
                    where: 
                    - \(X\) is the original value, 
                    - \(median(X)\) is the median of the feature,
                    - \(IQR\) is the interquartile range, calculated as \( Q3 - Q1 \) (the difference between the 75th and 25th percentiles).

                    This formula centers the data around the median and scales it according to the IQR, making it robust to outliers.
                    """)
                    scaled_robust = robust_scaling(df)
                    st.write("Scaled Data (Robust Scaling):")
                    scaled_robust.index = ['Value1', 'value2', 'Value3','value4','Value5','Value6','value7','Value8','value9','Value10']
                    st.write(scaled_robust.T)    

                    st.caption("To apply the Robust Scaling on our data we use sklearn library:")
                    st.code("""
                            from sklearn.preprocessing import RobustScaler
                            scaler = RobustScaler()
                            scaled = scaler.fit_transform(df)
                            """)

                    @st.experimental_dialog(" ", width="large")
                    def view_resources():
                        data = pd.read_csv('https://raw.githubusercontent.com/Nargis45/Feature_Engineering_Streamlit/main/st.csv')
                        data = data.head()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption('Before Robust Scaling')
                            st.dataframe(data, hide_index = True)
                            st.caption("Original Data Distribution")

                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            ax[0].hist(data['Age'], bins=10, color='skyblue', edgecolor='black')
                            ax[0].set_title('Age (Original)')
                            ax[1].hist(data['Fare'], bins=10, color='lightgreen', edgecolor='black')
                            ax[1].set_title('Fare (Original)')
                            st.pyplot(fig)
                        with col2:
                            sc = RobustScaler()
                            new_data = sc.fit_transform(data)
                            new_data = pd.DataFrame(new_data, columns=data.columns)
                            st.caption('After Robust Scaling')
                            st.dataframe(new_data, hide_index = True)

                            st.caption("Standardized Data Distribution")
                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            ax[0].hist(new_data['Age'], bins=10, color='skyblue', edgecolor='black')
                            ax[0].set_title('Age (Standardized)')
                            ax[1].hist(new_data['Fare'], bins=10, color='lightgreen', edgecolor='black')
                            ax[1].set_title('Fare (Standardized)')
                            st.pyplot(fig)
                        st.write("""
                        After Robust Scaling, there is no change in the distribution of data but look at scale of both `Age` and `Fare` features. 
                        """)

                    with stylable_container(
                        key="view",
                        css_styles="""
                        button {
                        background-color: #FF7F50;
                        color: white;
                    }
                    """
                    ):
                        if st.button('See Practical Example', key = 'Robust Scaler'):
                            view_resources()
                    st.write("""Effect on outliers:\n
                                Robust Scaling is specifically designed to be less sensitive to outliers because it uses the median and the interquartile range (IQR) instead of the mean and standard deviation. Outliers have less impact on the scaling.""")
                    st.write("""When to use:\n
                                Use when the data contains outliers, and you want a scaling technique that is robust and reduces their effect, such as when using algorithms that are sensitive to outliers (e.g., linear regression).""")
                
                st.divider()
                st.subheader("Difference betwwen Standardization and Normalization:")
                with st.expander("Open to see the difference"):
                    data = {
                        "Aspect": ["Definition", "Formula", "Scale", "Use cases", "Impact on Outliers"],
                        "Standardization": [
                            "Standardization (Z-score normalization) transforms the data to have a mean of 0 and a standard deviation of 1.",
                            "(X - mean) / std_dev",
                            "Data is centered around 0 with a standard deviation of 1. It does not bound the data within a specific range.",
                            "Used when data follows a Gaussian distribution, or when the machine learning algorithm assumes data is normally distributed (e.g., Logistic Regression, SVM).",
                            "Less sensitive to outliers. However, outliers can still affect the mean and standard deviation."
                        ],
                        "Normalization": [
                            "Normalization (Min-Max scaling) rescales the data to fit within a specific range, typically [0, 1].",
                            "(X - X_min) / (X_max - X_min)",
                            "Data is scaled to a fixed range, typically between 0 and 1.",
                            "Used when the algorithm requires data within a specific range (e.g., Neural Networks, KNN).",
                            "Highly sensitive to outliers, as they can skew the min and max values."
                        ]
                    }

                    # Convert the dictionary to a pandas DataFrame
                    df = pd.DataFrame(data)
                    st.table(df)

                st.divider()
                st.subheader("Practice Questions:")
                with st.expander("Feature Scaling Questions"):
                    questions = [
                        {
                            "question": "Which scaling technique should be used when the data contains significant outliers?",
                            "options": [
                                "Min-Max Scaling",
                                "Z-score Standardization",
                                "Robust Scaling",
                                "Log Transformation"
                            ],
                            "correct_answer": "Robust Scaling"
                        },
                        {
                            "question": "What is the purpose of Min-Max scaling?",
                            "options": [
                                "To make features normally distributed",
                                "To scale features to a range between 0 and 1",
                                "To handle missing values",
                                "To increase the variance"
                            ],
                            "correct_answer": "To scale features to a range between 0 and 1"
                        },
                        {
                            "question": "Which method is most commonly used to scale features when data follows a Gaussian distribution?",
                            "options": [
                                "Log Transformation",
                                "Z-score normalization",
                                "Min-Max scaling",
                                "Robust scaling"
                            ],
                            "correct_answer": "Z-score normalization"
                        },
                        {
                            "question": "What does Robust Scaling do?",
                            "options": [
                                "It removes outliers by clipping them",
                                "It scales data by removing outliers using median and interquartile range",
                                "It scales data by dividing by standard deviation",
                                "It centers data without scaling"
                            ],
                            "correct_answer": "It scales data by removing outliers using median and interquartile range"
                        },
                        {
                            "question": "When should you avoid using Min-Max scaling?",
                            "options": [
                                "When the data has outliers",
                                "When the data is normally distributed",
                                "When using decision tree-based algorithms",
                                "When using neural networks"
                            ],
                            "correct_answer": "When the data has outliers"
                        },
                        {
                            "question": "Which scaling technique is best for machine learning algorithms that assume normally distributed data?",
                            "options": [
                                "Min-Max Scaling",
                                "Log Transformation",
                                "Standardization (Z-score)",
                                "Robust Scaling"
                            ],
                            "correct_answer": "Standardization (Z-score)"
                        },
                        {
                            "question": "What is the key difference between Normalization and Standardization?",
                            "options": [
                                "Normalization scales data to a specific range, while standardization centers data to a mean of 0",
                                "Normalization removes outliers, while standardization does not",
                                "Normalization does not change the range of values, while standardization does",
                                "Standardization keeps the data in its original scale, while normalization changes the scale"
                            ],
                            "correct_answer": "Normalization scales data to a specific range, while standardization centers data to a mean of 0"
                        },
                        {
                            "question": "What happens if you apply standardization to data with outliers?",
                            "options": [
                                "It has no impact on the data",
                                "It will make the outliers more extreme",
                                "It reduces the impact of outliers by scaling them down",
                                "It increases the impact of outliers"
                            ],
                            "correct_answer": "It increases the impact of outliers"
                        },
                        {
                            "question": "What technique should be used when features have very different scales?",
                            "options": [
                                "Min-Max Scaling",
                                "Standardization",
                                "Robust Scaling",
                                "All of the above"
                            ],
                            "correct_answer": "All of the above"
                        },
                        {
                            "question": "Which feature scaling technique is particularly useful when features have varying scales and the model is sensitive to the magnitude of the features?",
                            "options": [
                                "Min-Max Scaling",
                                "Power Transformation",
                                "Z-score Standardization",
                                "Robust Scaling"
                            ],
                            "correct_answer": "Power Transformation"
                        }
                    ]

                    # Function to display the quiz
                    def display_quiz():
                        score = 0
                        responses = []
                        incorrect_answers = []

                        # Loop through each question and display it
                        for i, question in enumerate(questions):
                            st.caption(f"Question {i+1}:")
                            
                            # The key remains, but there's no locking of answers after selection
                            answer = st.radio(question["question"], options=question["options"], key=f"q{i}")

                            # Check the answer and store whether it's correct or not
                            if answer:
                                if answer == question["correct_answer"]:
                                    responses.append(True)
                                else:
                                    responses.append(False)
                                    incorrect_answers.append((i + 1, question["question"], answer, question["correct_answer"]))

                        # Submit button to calculate score
                        if st.button("Submit Test"):
                            score = sum(responses)
                            st.write(f"Your score is: {score}/10")
                            st.session_state["test_done"] = True

                            # Display incorrect answers with correct answers
                            if incorrect_answers:
                                st.write("You got the following questions wrong:")
                                for question_number, question_text, user_answer, correct_answer in incorrect_answers:
                                    st.divider()
                                    st.write(f"Question {question_number}: {question_text}")
                                    st.markdown(f"<span style='color:red'>Your Answer: {user_answer}</span>", unsafe_allow_html=True)
                                    st.markdown(f"<span style='color:green'>Correct Answer: {correct_answer}</span>", unsafe_allow_html=True)
                            else:
                                st.write("Great job! You answered all questions correctly.")

                        # # Button to retry the quiz
                        # if st.session_state.get("test_done", False):
                        #     if st.button("Try Again"):
                        #         # Reset the locked states only, not the answers
                        #         for i in range(10):
                        #             st.session_state[f"q{i}_locked"] = False
                        #         st.session_state["test_done"] = False
                        #         # Optionally reset answers if needed:  
                        #         for i in range(10):
                        #             st.session_state[f"q{i}"] = None

                    # Initialize session state for the first time
                    if "test_done" not in st.session_state:
                        st.session_state["test_done"] = False

                    display_quiz()



if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by Okld", page_icon="🔧", layout="wide")
    main()
