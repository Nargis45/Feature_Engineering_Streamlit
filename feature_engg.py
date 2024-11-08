import streamlit as st
from annotated_text import annotation
import annotated_text 

def main():
    button_labels = {
        "Home": "🏠 Home",
        "Feature Transformation": "🔄 Feature Transformation",
        "Feature Construction": "🔧 Feature Construction",
        "Feature Selection": "🔍 Feature Selection",
        "Feature Extraction": "📊 Feature Extraction"
    }

    # Initialize session state for all options
    if "selected_feature" not in st.session_state:
        st.session_state["selected_feature"] = "Home"  # Default to Home on first load

    # Sidebar for selecting features
    with st.sidebar:
        st.title("Feature Engineering")
        
        # Home button to reset selection to "Home"
        if st.button(button_labels["Home"]):
            st.session_state["selected_feature"] = "Home"

        # Expanders for each feature engineering option
        with st.expander("Explore Feature Engineering Options", expanded=True):
            for label, emoji_label in button_labels.items():
                if label != "Home":  # Exclude "Home" from the expander
                    if st.checkbox(emoji_label, key=label, value=(st.session_state["selected_feature"] == label)):
                        st.session_state["selected_feature"] = label

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

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by Okld", page_icon="🎈", layout="wide")
    main()
