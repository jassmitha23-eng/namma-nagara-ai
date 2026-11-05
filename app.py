import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Page Configuration
st.set_page_config(page_title="Namma Nagara AI", page_icon="🏙️", layout="wide")

st.title("🏙️ Namma Nagara AI")
st.subheader("One Karnataka. Smarter Cities. Better Living.")

# ==========================================
# THE AI BRAIN (Training the Model)
# ==========================================
# We give the AI examples of complaints and what category they belong to.
training_texts = [
    "There is a huge pile of garbage near the park", 
    "Trash is not being collected from my street",
    "The dustbin is overflowing",
    "Water is leaking from the main pipe",
    "No water supply in our area since morning",
    "The drainage is blocked and overflowing",
    "There is a big pothole on the main road",
    "The footpath is broken and dangerous",
    "Street lights are not working at night",
    "Power cut in our locality for 3 hours"
]

training_labels = [
    "Garbage", "Garbage", "Garbage", 
    "Water", "Water", "Water", 
    "Road", "Road", 
    "Electricity", "Electricity"
]

# Initialize and train the AI
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_texts)
model = MultinomialNB()
model.fit(X, training_labels)

# (AI is now trained and ready!)

# ==========================================
# DATABASE SETUP
# ==========================================
DATA_FILE = "complaints.csv"
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=["ID", "Citizen Name", "Complaint Text", "Location", "AI Category", "Status"])
    df.to_csv(DATA_FILE, index=False)

# Tabs
tab1, tab2 = st.tabs(["📝 File a Complaint", "🏢 Official Dashboard"])

# ==========================================
# TAB 1: CITIZEN PORTAL
# ==========================================
with tab1:
    st.header("File a Civic Complaint")
    
    with st.form("complaint_form"):
        name = st.text_input("Your Name")
        location = st.text_input("Location (e.g., Ward 42, Jayanagar)")
        complaint_text = st.text_area("Describe the issue (e.g., 'There is a huge pile of garbage near the park')")
        
        submitted = st.form_submit_button("Submit Complaint")
        
        if submitted:
            if name and location and complaint_text:
                # 🧠 USE THE AI TO PREDICT THE CATEGORY
                new_text_vect = vectorizer.transform([complaint_text])
                predicted_category = model.predict(new_text_vect)[0]
                
                df = pd.read_csv(DATA_FILE)
                new_id = len(df) + 1
                
                new_complaint = pd.DataFrame([{
                    "ID": new_id, 
                    "Citizen Name": name, 
                    "Complaint Text": complaint_text, 
                    "Location": location, 
                    "AI Category": predicted_category, # AI prediction goes here!
                    "Status": "Filed"
                }])
                
                df = pd.concat([df, new_complaint], ignore_index=True)
                df.to_csv(DATA_FILE, index=False)
                
                st.success(f"✅ Complaint filed successfully! Our AI categorized this as: **{predicted_category}**")
            else:
                st.error("Please fill in all fields.")

# ==========================================
# TAB 2: OFFICIAL DASHBOARD
# ==========================================
with tab2:
    st.header("Municipal Dashboard")
    
    df = pd.read_csv(DATA_FILE)
    
    if len(df) == 0:
        st.info("No complaints filed yet.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Total Complaints", len(df))
        col2.metric("Resolved", len(df[df["Status"] == "Resolved"]))
        
        st.write("### All Complaints (AI Categorized)")
        
        st.dataframe(df, width="stretch")