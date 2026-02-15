import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide")
st.title("🎲 Baccarat AI Pro Cloud Edition")

# Database Setup
conn = sqlite3.connect("baccarat_data.db")
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS history
             (result TEXT)""")
conn.commit()

# Add Result Buttons
col1, col2 = st.columns(2)

if col1.button("Banker (B)"):
    c.execute("INSERT INTO history VALUES ('B')")
    conn.commit()

if col2.button("Player (P)"):
    c.execute("INSERT INTO history VALUES ('P')")
    conn.commit()

# Load History
data = pd.read_sql("SELECT * FROM history", conn)
history = data['result'].tolist()

st.write("Current Shoe:", history)

# AI Section
window = 10

if len(history) > window:
    encoded = [1 if x=='B' else 0 for x in history]

    X, y = [], []
    for i in range(len(encoded)-window):
        X.append(encoded[i:i+window])
        y.append(encoded[i+window])

    X = np.array(X)
    y = np.array(y)

    # ML Model
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    last = np.array(encoded[-window:]).reshape(1,-1)
    prob = model.predict_proba(last)[0][1]

    prediction = "Banker" if prob > 0.5 else "Player"
    confidence = abs(prob-0.5)*200

    st.subheader("🎯 AI Prediction")
    st.write("Prediction:", prediction)
    st.write("Probability %:", round(prob*100,2))
    st.write("Confidence %:", round(confidence,2))

    # Kelly Bet
    bankroll = st.number_input("Enter Bankroll", value=10000)
    b = 0.95
    kelly = (prob*(b+1)-1)/b
    bet = max(0, kelly)*bankroll

    st.subheader("💰 Suggested Bet")
    st.write(round(bet,2))

# Reset Shoe
if st.button("Reset Shoe"):
    c.execute("DELETE FROM history")
    conn.commit()
    st.experimental_rerun()
