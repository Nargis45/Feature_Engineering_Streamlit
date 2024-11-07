import streamlit as st
from streamlit_navigation_bar import st_navbar

page = st_navbar(["Home", "Feature Transformation", "Feature Construction", "Featue Scaling", "Feature Extraction"])
st.write(page)
