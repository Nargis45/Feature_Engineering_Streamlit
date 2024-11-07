import streamlit as st
from streamlit_gallery import apps
from streamlit_gallery.utils.page import page_group

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
        st.success("1. Feature Transformation")
    with st.expander("2. Feature Construction"):
        st.info("2. Feature Construction")
    with st.expander("3. Feature Selection"):
        st.warning("3. Feature Selection")
    with st.expander("4. Feature Extraction"):
        st.error("4. Feature Extraction")

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by Okld", page_icon="🎈", layout="wide")
    main()
