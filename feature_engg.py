import streamlit as st
from annotated_text import annotation
import annotated_text 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_extras.stylable_container import stylable_container
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

def main():
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
                                1. No effect on outliers\n
    2. Outliers effect do not get reduced""")
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
