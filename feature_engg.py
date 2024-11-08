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
    if 'selected_feature' not in st.session_state:
        st.session_state.selected_feature = "Home"
    if 'selected_sub_option' not in st.session_state:
        st.session_state.selected_sub_option = None

    def select_feature(label, sub_option=None):
        # Set the selected feature and optionally the sub-option
        st.session_state.selected_feature = label
        st.session_state.selected_sub_option = sub_option

    with st.sidebar:
        st.title("Feature Engineering")

        # Home button
        if st.button(sidebar_labels["Home"]):
            select_feature("Home")

        # Expanders with radio buttons for each feature category
        for label in list(sidebar_labels.keys())[1:]:
            with st.expander(sidebar_labels[label], expanded=(st.session_state.selected_feature == label)):
                selected_option = st.radio(
                    f"Select an option for {label}",
                    options=[f"{label} - Option 1", f"{label} - Option 2", f"{label} - Option 3"],
                    key=f"radio_{label}"
                )
                # Update selected feature and sub-option when a radio button is selected
                if selected_option:
                    select_feature(label, selected_option)

    # Main content based on the selected sidebar option
    if st.session_state.selected_feature == "Home":
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
        selected_feature = st.session_state.selected_feature
        selected_option = st.session_state.selected_sub_option or "No option selected"
        st.write(f"**Selected Technique:** {selected_feature}")
        st.write(f"**Option:** {selected_option}")

if __name__ == "__main__":
    st.set_page_config(page_title="Feature Engineering App", page_icon="🔧", layout="wide")
    main()
