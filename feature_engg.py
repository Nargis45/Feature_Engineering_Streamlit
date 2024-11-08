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
    
    def select_feature(label, sub_option=None):
        # Reset all main feature states to False, then set the selected feature state to True
        for key in sidebar_labels.keys():
            st.session_state[key] = (key == label)

        # Set selected sub-option if available
        if sub_option:
            st.session_state['selected_sub_option'] = sub_option

    with st.sidebar:
        st.title("Feature Engineering")

        # Home button
        if st.button(sidebar_labels["Home"]):
            select_feature("Home")

        # Expanders with radio buttons for each feature category
        for label in list(sidebar_labels.keys())[1:]:
            with st.expander(sidebar_labels[label]):
                selected_option = None
                if st.session_state[label]:  # Render radio only when a category is selected
                    selected_option = st.radio(
                        f"Select an option for {label}",
                        options=[f"{label} - Option 1", f"{label} - Option 2", f"{label} - Option 3"],
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
        for label in list(sidebar_labels.keys())[1:]:
            if st.session_state[label]:
                st.write(f"**Selected Technique:** {label}")
                st.write(f"**Option:** {st.session_state.get('selected_sub_option', 'No option selected')}")
                break

if __name__ == "__main__":
    st.set_page_config(page_title="Feature Engineering App", page_icon="🔧", layout="wide")
    main()
