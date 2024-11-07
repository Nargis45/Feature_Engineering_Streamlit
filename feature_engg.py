import streamlit as st
from streamlit_gallery import apps
from streamlit_gallery.utils.page import page_group
from annotated_text import annotation
import annotated_text 

def main():
    page = page_group("p")

    button_labels = {
            "Feature Transformation": "🔄 Feature Transformation",
            "Feature Construction": "🔧 Feature Construction",
            "Feature Selection": "🔍 Feature Selection",
            "Feature Extraction": "📊 Feature Extraction"
        }

    for label in button_labels.keys():
        if label not in st.session_state:
            st.session_state[label] = False

    with st.sidebar:
        st.title("Feature Engineering")

        with st.expander(" ", True):
        # Define each button with equal size
        cols = st.columns(1)
        for label, emoji_label in button_labels.items():
            if cols[0].button(emoji_label, key=label):
                # Reset all buttons, then set the clicked button to True
                for key in button_labels.keys():
                    st.session_state[key] = (key == label)

    for label in button_labels.keys():
        if st.session_state[label]:
            st.write(f"**Selected Technique:** {label}")
            break

        # with st.expander("✨ APPS", True):
        #     page.item("Streamlit gallery", apps.gallery, default=True)

        # with st.expander(" ", True):
        #     st.button("Feature Transformation")
        #     st.button("Feature Construction")
        #     st.button("Feature Selection")
        #     st.button("Feature Extraction")
        

    st.subheader("Feature Engineering is the process of using domain knowledge to extract features from raw data. Those features can be used to improve the performance of machine learning algorithms.")
    st.title("Types of Feature Engineering:")
    with st.expander("1. Feature Transformation"):
        st.success("Feature Transformation is a technique used to transform a feature/column by applying mathematical formula which is going to be useful for further analysis.")
    with st.expander("2. Feature Construction"):
        st.info("Feature Construction is creating new feature using existing features.")
    with st.expander("3. Feature Selection"):
        st.warning("Feature Selection is selecting important features from the given features to improve model performance.")
    with st.expander("4. Feature Extraction"):
        st.error("Feature Extraction is creating completely new features out of the given features.")

    annotated_text.annotated_text( 
            annotation("Click on the sidebar to try the practical implementation of each type", color='#07a631'),
        )

    # st.caption("Click on the sidebar to try the practical implementation of each type")

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by Okld", page_icon="🎈", layout="wide")
    main()
