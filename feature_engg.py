import streamlit as st
from annotated_text import annotation
import annotated_text

def main():
    # Define a dictionary for the sidebar labels
    sidebar_labels = {
        "Home": "🏠 Home",
        "Feature Transformation": "🔄 Feature Transformation",
        "Feature Construction": "🔧 Feature Construction",
        "Feature Selection": "🔍 Feature Selection",
        "Feature Extraction": "📊 Feature Extraction"
    }
    
    # Initialize session states for sidebar items
    for label in sidebar_labels.keys():
        if label not in st.session_state:
            st.session_state[label] = (label == "Home")

    if 'selected_sub_option' not in st.session_state:
        st.session_state['selected_sub_option'] = None
    
    def select_feature(label, sub_option=None):
        # Reset all main feature states to False, then set the selected feature state to True
        for key in sidebar_labels.keys():
            st.session_state[key] = (key == label)

        # Set selected sub-option if available
        st.session_state['selected_sub_option'] = sub_option

    with st.sidebar:
        st.title("Feature Engineering")

        # Home button
        if st.button(sidebar_labels["Home"]):
            select_feature("Home")

        # Expanders with radio buttons for each feature category
        for label in list(sidebar_labels.keys())[1:]:
            with st.expander(sidebar_labels[label]):
                if st.session_state[label]:  # Render radio buttons only when a category is selected
                    selected_option = st.radio(
                        f"Select an option for {label}",
                        options=[
                            f"{label} - Option 1: Overview",
                            f"{label} - Option 2: Use Cases",
                            f"{label} - Option 3: Best Practices"
                        ],
                        key=label
                    )
                    if selected_option:
                        select_feature(label, selected_option)
    
    # Main content based on the selected sidebar option
    if st.session_state["Home"]:
        st.subheader("Feature Engineering is the process of using domain knowledge to extract features from raw data. These features can improve machine learning algorithms.")
        st.title("Types of Feature Engineering:")
        
        with st.expander("1. Feature Transformation"):
            st.success("Feature Transformation applies mathematical transformations to a feature, making it more suitable for analysis.")
            st.success("""
                Methods for Feature Transformation include:
                - Missing value imputation
                - Handling categorical values
                - Outlier detection
                - Feature scaling
            """)
        
        with st.expander("2. Feature Construction"):
            st.info("Feature Construction involves creating new features from existing ones.")
        
        with st.expander("3. Feature Selection"):
            st.warning("Feature Selection identifies key features to improve model performance.")
        
        with st.expander("4. Feature Extraction"):
            st.error("Feature Extraction generates new features based on existing data.")
        
        annotated_text.annotated_text(
            annotation("Use the sidebar to explore each feature type.", color='#07a631')
        )

    else:
        # Content for each feature type and selected option
        if st.session_state['selected_sub_option']:
            selected_option = st.session_state['selected_sub_option']
            if "Feature Transformation" in selected_option:
                st.write(f"**Selected Technique:** Feature Transformation")
                if "Overview" in selected_option:
                    st.write("Feature Transformation involves applying mathematical transformations to modify features for better model performance.")
                elif "Use Cases" in selected_option:
                    st.write("Common use cases include scaling features, encoding categorical variables, and imputing missing values.")
                elif "Best Practices" in selected_option:
                    st.write("Use transformations that maintain the interpretability of features. Always normalize or standardize when using distance-based models.")
            
            elif "Feature Construction" in selected_option:
                st.write(f"**Selected Technique:** Feature Construction")
                if "Overview" in selected_option:
                    st.write("Feature Construction is creating new features from existing ones to capture more information.")
                elif "Use Cases" in selected_option:
                    st.write("Examples include generating interaction terms, polynomial features, or creating time-based features.")
                elif "Best Practices" in selected_option:
                    st.write("Construct features that are meaningful and reduce dimensionality where possible. Avoid excessive feature generation to prevent overfitting.")
            
            elif "Feature Selection" in selected_option:
                st.write(f"**Selected Technique:** Feature Selection")
                if "Overview" in selected_option:
                    st.write("Feature Selection involves selecting the most relevant features for model training.")
                elif "Use Cases" in selected_option:
                    st.write("Use feature selection to reduce dimensionality and improve model performance by retaining only important features.")
                elif "Best Practices" in selected_option:
                    st.write("Consider domain knowledge when selecting features. Use techniques like correlation analysis, feature importance scores, or L1 regularization.")

            elif "Feature Extraction" in selected_option:
                st.write(f"**Selected Technique:** Feature Extraction")
                if "Overview" in selected_option:
                    st.write("Feature Extraction generates new, informative features by transforming existing ones.")
                elif "Use Cases" in selected_option:
                    st.write("Common methods include Principal Component Analysis (PCA) and Latent Dirichlet Allocation (LDA) for dimensionality reduction.")
                elif "Best Practices" in selected_option:
                    st.write("Use extraction techniques that improve model interpretability. Ensure extracted features retain significant information from original data.")

if __name__ == "__main__":
    st.set_page_config(page_title="Feature Engineering App", page_icon="🔧", layout="wide")
    main()
