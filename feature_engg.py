import streamlit as st
from annotated_text import annotation
import annotated_text 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from streamlit_extras.stylable_container import stylable_container
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

def main():
              list_tabs = ["Encoding Techniques", "Feature Scaling"]
              tab1, tab2 = st.tabs(list_tabs)
              with tab1:
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
                def one_hot_encode(df, column):
                    ohe = pd.get_dummies(df[column], drop_first=True)
                    df = df.join(ohe)
                    df.drop(column, axis=1, inplace=True)
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
              with tab2:
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
