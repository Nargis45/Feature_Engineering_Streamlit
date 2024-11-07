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
            page.item("Ace editor")
            page.item("Disqus")
            page.item("Elements⭐")
            page.item("Pandas profiling")
            page.item("Quill editor")
            page.item("React player")

    page.show()

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by Okld", page_icon="🎈", layout="wide")
    main()
