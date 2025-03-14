import streamlit as st

def section_separator():
    """Display a horizontal line separator"""
    st.markdown("---")

def get_assessment_components(data):
    """
    Identify assessment components dynamically from the data
    
    Parameters:
    - data: DataFrame containing student data
    
    Returns:
    - List of column names corresponding to assessment components
    """
    exclude_cols = ['Student_ID', 'Final Mark', 'Course', 'Cluster']
    primary_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('Essay')]
    return primary_cols