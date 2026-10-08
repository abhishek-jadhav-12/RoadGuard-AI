import streamlit as st


st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🚨",
    layout="wide",
)


st.title("RoadGuard AI")
st.subheader("Real-Time Road Accident & Emergency Intelligence System")

st.info(
    "Dashboard foundation is ready. "
    "Computer vision and accident detection modules will be integrated next."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Vehicles", "0")

with col2:
    st.metric("Active Incidents", "0")

with col3:
    st.metric("High Severity", "0")

with col4:
    st.metric("System Status", "ONLINE")