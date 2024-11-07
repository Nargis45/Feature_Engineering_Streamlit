import streamlit as st
from streamlit_navigation_bar import st_navbar

from streamlit_gallery import apps
from streamlit_gallery.utils.page import page_group

def main():
    page = page_group("p")

    with st.sidebar:
        st.title("🎈 Okld's Gallery")

        with st.expander("✨ APPS", True):
            page.write("Streamlit gallery")

        with st.expander("🧩 COMPONENTS", True):
            st.button("Ace editor")
            st.button("Disqus")
            st.button("Elements⭐")
            st.button("Pandas profiling")
            st.button("Quill editor")
            st.button("React player")

    page.show()

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by Okld", page_icon="🎈", layout="wide")
    main()
