import streamlit as st
from annotated_text import annotation
import annotated_text 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_extras.stylable_container import stylable_container

def main():
    button_labels = {
        "Home": "🏠 Home",
        "Feature Transformation": "🔄 Feature Transformation",
        "Feature Construction": "🔧 Feature Construction",
        "Feature Selection": "🔍 Feature Selection",
        "Feature Extraction": "📊 Feature Extraction"
    }

    # Initialize session state for selected feature
    if "selected_feature" not in st.session_state:
        st.session_state["selected_feature"] = "Home"  # Default to Home on first load

    # Sidebar for selecting features
    with st.sidebar:
        st.title("Feature Engineering")

        # Radio button for feature selection
        selected_option = st.radio(
            "Choose a feature engineering technique:",
            options=list(button_labels.keys()),
            format_func=lambda x: button_labels[x]
        )
        
        # Update session state when the selected option changes
        st.session_state["selected_feature"] = selected_option

    # Display content based on selected option in the sidebar
    if st.session_state["selected_feature"] == "Home":
        st.subheader("Feature Engineering is the process of using domain knowledge to extract features from raw data. Those features can be used to improve the performance of machine learning algorithms.")
        st.title("Types of Feature Engineering:")
        with st.expander("1. Feature Transformation"):
            st.success("Feature Transformation is a technique used to transform a feature/column by applying mathematical formula which is going to be useful for further analysis.")
            st.success("""Methods we can use in Feature Transformation:\n
                            - Missing value imputation\n
    - Handling categorical values\n
    - Outlier detection\n
    - Feature scaling""")
        with st.expander("2. Feature Construction"):
            st.info("Feature Construction is creating new feature using existing features.")
        with st.expander("3. Feature Selection"):
            st.warning("Feature Selection is selecting important features from the given features to improve model performance.")
        with st.expander("4. Feature Extraction"):
            st.error("Feature Extraction is creating completely new features out of the given features.")
        
        annotated_text.annotated_text(
            annotation("Use the sidebar to explore each feature type.", color='#07a631')
        )
    else:
        # Display the selected technique when not on "Home"
        selected_feature = st.session_state["selected_feature"]
        st.write(f"**Selected Technique:** {selected_feature}")
        if selected_feature == "Feature Transformation":
            list_tabs = ["Missing Value Imputation", "Handling Categorical Values", "Outlier Detection", "Feature Scaling"]
            tab1, tab2, tab3, tab4= st.tabs(list_tabs)
            with tab1:
                pass
            with tab2:
                pass
            with tab3:
                pass
            with tab4:
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
                        Standardization (also known as Z-score Normalization) is a technique that transforms data to have a mean of 0 and a standard deviation of 1. This process adjusts each feature so that it follows a standard normal distribution (or approximately normal), which is useful for many machine learning algorithms that assume normally distributed data, such as linear regression and logistic regression.
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

                    # Mean-centered and standardized data
                    standardized_data = (data - mean) / std_dev

                    # Plot original, mean-centered, and standardized data
                    fig, ax = plt.subplots(figsize=(5, 3))  # Smaller figure size

                    ax.plot(data, label='Original Data', marker='o', linestyle='-', color='b', markersize=4, linewidth=1)
                    ax.plot(standardized_data, label='Standardized Data', marker='o', linestyle=':', color='g', markersize=4, linewidth=1)

                    ax.set_xlabel('Index', fontsize=8)
                    ax.set_ylabel('Value', fontsize=8)
                    ax.set_title('Standardization of Data', fontsize=10)
                    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))  # Position legend outside plot for space efficiency

                    plt.tight_layout()

                    st.pyplot(fig)

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
                    @st.experimental_dialog(" ", width="large")
                    def view_resources():
                         pass 
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

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by Okld", page_icon="🎈", layout="wide")
    main()
