import streamlit as st
from streamlit_gallery import apps
from streamlit_gallery.utils.page import page_group
from annotated_text import annotation
import annotated_text 

def main():
    page = page_group("p")

    with st.sidebar:
        st.title("Feature Engineering")

        # with st.expander("✨ APPS", True):
        #     page.item("Streamlit gallery", apps.gallery, default=True)

        with st.expander(" ", True):
            st.button("Feature Transformation")
            st.button("Feature Construction")
            st.button("Feature Selection")
            st.button("Feature Extraction")

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
            annotation("Click on the sidebar to try the practical implementation of each type", border='3px groove yellow')
        )

    # st.caption("Click on the sidebar to try the practical implementation of each type")

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by Okld", page_icon="🎈", layout="wide")
    main()
