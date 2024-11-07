import streamlit as st
from streamlit_navigation_bar import st_navbar

from streamlit_gallery import apps
from streamlit_gallery.utils.page import page_group

def main():
    page = page_group("p")

    with st.sidebar:
        st.title("🎈 Okld's Gallery")

        with st.expander("✨ APPS", True):
            page.item("Streamlit gallery", apps.gallery, default=True)

        with st.expander("🧩 COMPONENTS", True):
            st.write("Ace editor")
            st.write("Disqus")
            st.write("Elements⭐")
            st.write("Pandas profiling")
            st.write("Quill editor")
            st.write("React player")

    page.show()

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by Okld", page_icon="🎈", layout="wide")
    main()
