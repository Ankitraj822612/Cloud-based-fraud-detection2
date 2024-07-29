import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import streamlit as st
from datetime import datetime

# Load the data with error handling
try:
    data = pd.read_csv('creditcard.csv', on_bad_lines='skip')
except pd.errors.ParserError as e:
    st.write("Error reading the CSV file: ", e)

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Balance the dataset
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Features and target
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy scores
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

st.title("Credit Card Fraud Detection Model")
st.write("Enter the following details to check if the transaction is legitimate or fraudulent:")
st.image("https://www.mwanmobile.com/wp-content/uploads/2022/11/1.jpg")

# Streamlit inputs
card_number = st.text_input('Credit Card Number: ')
transaction_time = st.text_input('Transaction Time : ')
transaction_id = st.text_input('Transaction Id: ')
amount = st.text_input('Amount: ')
location = st.text_input('Location of the Transaction: ')
cardholder_name = st.text_input('Cardholder Name: ')
cvv_code = st.text_input('CVV Code: ')
issuing_bank = st.text_input('Issuing Bank: ')
merchant_name = st.text_input('Merchant Name: ')
transaction_type = st.selectbox('Transaction Type:', ['Online', 'In-Store', 'Mobile', 'ATM'])


submit = st.button("Submit")
accuracy = st.button("Accuracy")

# Prediction and new data entry
if submit:
    try:
        # Validate Credit Card Number
        if len(card_number) != 16 or not card_number.isdigit():
            st.write("Please enter a valid 16-digit credit card number.")
            raise ValueError("Invalid credit card number")

        # Validate Transaction ID
        if len(transaction_id) != 8 or not transaction_id.isdigit():
            st.write("Please enter a valid 8-digit transaction ID.")
            raise ValueError("Invalid transaction ID")

        # Validate Transaction Time Format
        try:
            time_obj = datetime.strptime(transaction_time, '%H:%M')
            transaction_time_seconds = time_obj.hour * 3600 + time_obj.minute * 60
        except ValueError:
            st.write("Invalid time format. Please enter in HH:MM format.")
            raise ValueError("Invalid time format. Please enter in HH:MM format.")

        # Validate Amount
        amount = float(amount)

        # Adjust the number of features to match the training data
        features = np.zeros(X_train.shape[1])
        
        # Here we assume the first column is 'Time' and the last column is 'Amount'
        features[0] = transaction_time_seconds
        features[-1] = amount 
        
        prediction = model.predict(features.reshape(1, -1))

        if prediction[0] == 0:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")

        new_data = {
            'Time': transaction_time_seconds,
            'Amount': amount,
            'Class': prediction[0],
            **{f'V{i}': features[i] for i in range(1, len(features) - 1)}  # Adjusted to correctly index 'V' features
        }

        # Additional information to be saved but not used for prediction
        additional_info = {
            'Credit Card Number': card_number,
            'Transaction Id': transaction_id,
            'Location of the Transaction': location,
            'Cardholder Name': cardholder_name,
            'CVV Code': cvv_code,
            'Issuing Bank': issuing_bank,
            'Merchant Name': merchant_name,
            'Transaction Type': transaction_type,
            
        }

        new_data.update(additional_info)
        new_data_df = pd.DataFrame(new_data, index=[0])
        new_data_df.to_csv('creditcard.csv', mode='a', header=False, index=False)

    except ValueError as e:
        st.write(str(e))

# Display accuracy
if accuracy:
    st.write(f"Accuracy on Training data: {train_acc * 100:.2f}%")
    st.write(f"Accuracy on Test data: {test_acc * 100:.2f}%")
