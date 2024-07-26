import streamlit as st

st.write("Hi streamlit")
st.subheader("Hello World!")
st.selectbox("Which programming language you like?",['Python', 'Java', 'C++'])
st.checkbox("Python")
st.slider("Pick some value", 0, 100)
st.select_slider("Select entry", ["Best", "Avg", "Worst"])
st.progress(10)
st.button("Enter")

# sidebar
st.sidebar.title("About")
st.sidebar.selectbox("wht would u like to see?", ["home", "about", "contact us"])
st.sidebar.markdown('info')
st.sidebar.button("Submit")
