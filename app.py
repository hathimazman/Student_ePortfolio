import streamlit as st
import pandas as pd
import numpy as np
import os
from student_analysis2 import analyze_student_performance, analyze_multiple_courses

# Import visualization modules
from tabs.overview_tab import render_overview_tab
from tabs.intervention_tab import render_intervention_tab
from tabs.clusters_tab import render_clusters_tab
from tabs.components_tab import render_components_tab
from tabs.portfolio_tab import render_portfolio_tab
from utils.helper_functions import section_separator, get_assessment_components

# Set page config
st.set_page_config(
    page_title="Student Performance Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'course_files' not in st.session_state:
    st.session_state.course_files = {}  # Dictionary to store course files
if 'results' not in st.session_state:
    st.session_state.results = None  # To store analysis results
if 'selected_courses' not in st.session_state:
    st.session_state.selected_courses = []  # To store selected courses for analysis
if 'analyze_button_clicked' not in st.session_state:
    st.session_state.analyze_button_clicked = False

# App title
st.title("Medical Education Dashboard with Artificial Intelligence [MED-AI]")

# Sidebar for file uploads and course selection
with st.sidebar:
    st.header("Course Data")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a course CSV file", type="csv", key="file_upload")
    
    if uploaded_file is not None:
        # Get course name from user
        course_name = st.text_input("Enter a name for this course:", f"Course {len(st.session_state.course_files) + 1}")
        
        # Add button to add the course
        if st.button("Add Course"):
            # Save the file to a temporary location
            temp_file_path = f"temp_{course_name.replace(' ', '_')}.csv"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Add to course files dictionary
            st.session_state.course_files[course_name] = temp_file_path
            st.success(f"Added {course_name} to courses!")
    
    # Display list of added courses
    if st.session_state.course_files:
        st.subheader("Added Courses")
        for i, (course, _) in enumerate(st.session_state.course_files.items()):
            st.write(f"{i+1}. {course}")
        
        # Option to remove courses
        course_to_remove = st.selectbox("Select course to remove:", 
                                      ["None"] + list(st.session_state.course_files.keys()),
                                      key="remove_course")
        
        if course_to_remove != "None" and st.button("Remove Course"):
            # Delete the temporary file
            if os.path.exists(st.session_state.course_files[course_to_remove]):
                os.remove(st.session_state.course_files[course_to_remove])
            
            # Remove from dictionary
            del st.session_state.course_files[course_to_remove]
            st.success(f"Removed {course_to_remove}!")
            st.rerun()  # Rerun the app to update the sidebar
    
    # Multiselect for courses to analyze
    if st.session_state.course_files:
        st.subheader("Select Courses to Analyze")
        st.session_state.selected_courses = st.multiselect(
            "Choose courses",
            options=list(st.session_state.course_files.keys()),
            default=list(st.session_state.course_files.keys()),
            key="course_selector"
        )
        
        # Button to run analysis
        if st.button("Analyze Selected Courses"):
            with st.spinner("Running analysis..."):
                # Create a dictionary with only the selected courses
                selected_files = {course: st.session_state.course_files[course] 
                                for course in st.session_state.selected_courses}
                
                # Run the analysis for selected courses
                st.session_state.results = analyze_multiple_courses(selected_files)
                st.session_state.analyze_button_clicked = True
            
            st.success("Analysis complete!")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This dashboard analyzes student performance across multiple courses. "
        "Upload CSV files containing student assessment data to generate insights "
        "on at-risk students, performance clusters, and assessment components."
    )

# Main content area - only show if analysis has been run
if st.session_state.analyze_button_clicked and st.session_state.results:
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Early Intervention", "Student Clusters", "Component Analysis", "Student e-Portfolio"])
    
    # Get all course names including "All Courses" if it exists
    all_course_names = list(st.session_state.results.keys())
    
    # Tab 1: Overview
    with tab1:
        render_overview_tab(st.session_state.results, all_course_names, section_separator, get_assessment_components)
    
    # Tab 2: Early Intervention
    with tab2:
        render_intervention_tab(st.session_state.results, all_course_names, section_separator, get_assessment_components)
    
    # Tab 3: Student Clusters
    with tab3:
        render_clusters_tab(st.session_state.results, all_course_names, section_separator, get_assessment_components)
    
    # Tab 4: Component Analysis
    with tab4:
        render_components_tab(st.session_state.results, all_course_names, section_separator)
    
    # Tab 5: Student e-Portfolio
    with tab5:
        render_portfolio_tab(st.session_state.results, all_course_names, section_separator, get_assessment_components)
else:
    # Display initial instructions if no analysis has been run
    st.info(
        """
        ### Welcome to the MED-AI (Medical Education Dashboard with Artificial Intelligence)!
        
        To get started:
        1. Use the sidebar to upload CSV files containing your course data
        2. Name each course and click "Add Course"
        3. Once you've added all courses, select the ones you want to analyze
        4. Click "Analyze Selected Courses" to run the analysis
        
        Your CSV files should have the following structure:
        - Student_ID: Student identifier
        - Assessment components (MCQ, MEQ, OSPE, PAM, PBL, etc.) : Ensure that your marks have been calculated (divide by total marks, times by weightage)
        - Final Mark: Overall final mark
        
        The analysis will provide insights on student intervention needs, performance clusters, and assessment components.
        """
    )

    # Sample CSV structure
    st.subheader("Example CSV Structure")
    sample_data = pd.DataFrame({
        'Student_ID': ['A161342', 'A152487', 'A178642'],
        'MCQ': [15.5, 18.2, 12.7],
        'MEQ': [20.1, 19.5, 17.8],
        'OSPE': [6.5, 7.8, 5.9],
        'PAM': [12.5, 14.2, 11.8],
        'PBL': [8.5, 9.2, 7.8],
        'Final Mark': [63.1, 68.9, 56]
    })
    st.dataframe(sample_data)
