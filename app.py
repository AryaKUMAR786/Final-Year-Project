from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///investment.db'

db = SQLAlchemy(app)
# important app password for startup funding : imdh soqg ncim nwaf

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'ayushmanaryak@gmail.com'  # Sender email
app.config['MAIL_PASSWORD'] = 'imdhsoqgncimnwaf'  # App password
mail = Mail(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)  # Store email
    password = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'client' or 'investor'

# Load dataset
data = pd.read_csv("data.csv")

def preprocess_data(df):
    df['Amount in USD'] = df['Amount in USD'].str.replace(',', '', regex=True)
    df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')
    df = df.dropna(subset=['Amount in USD'])
    
    categorical_cols = ['Industry Vertical', 'Sub Vertical', 'City  Location', 'Investment Type']
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(df[categorical_cols]).toarray()
    
    scaler = StandardScaler()
    scaled_amount = scaler.fit_transform(df[['Amount in USD']])
    
    return df, encoded_features, encoder, scaler, scaled_amount

def train_model(df, encoded_features, scaled_amount):
    X = np.hstack((encoded_features, scaled_amount))
    
    model = NearestNeighbors(n_neighbors=15, metric='euclidean')
    model.fit(X)
    return model

data, encoded_data, encoder, scaler, scaled_amount = preprocess_data(data)
model = train_model(data, encoded_data, scaled_amount)

pending_requests = []  # Store client investment requests
investor_dashboard = {}  # Store investor-wise investment requests

def send_email(subject, recipient, message):
    msg = Message(subject, sender='your-email@gmail.com', recipients=[recipient])
    msg.body = message
    mail.send(msg)

def recommend_investors(query):
    query_df = pd.DataFrame([query])
    encoded_query = encoder.transform(query_df[['Industry Vertical', 'Sub Vertical', 'City  Location', 'Investment Type']]).toarray()
    scaled_query_amount = scaler.transform([[query['Amount in USD']]])
    query_vector = np.hstack((encoded_query, scaled_query_amount))
    
    distances, indices = model.kneighbors(query_vector, n_neighbors=15)
    
    recommendations = []
    for idx in indices[0]:
        if data.iloc[idx]['Amount in USD'] >= query['Amount in USD']:
            investor = data.iloc[idx]['Investors Name']
            recommendations.append(investor)
            if investor not in investor_dashboard:
                investor_dashboard[investor] = []
            investor_dashboard[investor].append(query)
        if len(recommendations) == 5:
            break
    
    return recommendations

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        role = request.form['role']
        
        new_user = User(username=username, email=email, password=password, role=role)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None  # To store error messages
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user'] = username
            session['role'] = user.role
            return redirect(url_for('client_dashboard') if user.role == 'client' else url_for('investor_dashboard_view', name=username))
        else:
            error = "Invalid username or password."

    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('role', None)
    return redirect(url_for('home'))

@app.route('/client', methods=['GET', 'POST'])
def client_dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    client = User.query.filter_by(username=session['user']).first()
    
    if request.method == 'POST':
        query = {
            "Industry Vertical": request.form['industry'],
            "Sub Vertical": request.form['sub_vertical'],
            "City  Location": request.form['city'],
            "Investment Type": request.form['investment_type'],
            "Amount in USD": float(request.form['amount']),
            "Client Name": session['user'],
            "Client Email": client.email
        }
        investors = recommend_investors(query)
        pending_requests.append(query)
        return render_template('client.html', investors=investors)
    
    return render_template('client.html', investors=[])

@app.route('/investor/<name>')
def investor_dashboard_view(name):
    investments = investor_dashboard.get(name, [])
    return render_template('investor.html', investor=name, investments=investments)

@app.route('/accept/<name>/<int:index>')
def accept_investment(name, index):
    if name in investor_dashboard and index < len(investor_dashboard[name]):
        request_data = investor_dashboard[name].pop(index)
        send_email("Investment Request Accepted", request_data['Client Email'], 
                   f"Dear {request_data['Client Name']},\n\nYour investment request has been accepted by {name}.")
    return redirect(url_for('investor_dashboard_view', name=name))

@app.route('/decline/<name>/<int:index>')
def decline_investment(name, index):
    if name in investor_dashboard and index < len(investor_dashboard[name]):
        request_data = investor_dashboard[name].pop(index)
        send_email("Investment Request Declined", request_data['Client Email'], 
                   f"Dear {request_data['Client Name']},\n\nUnfortunately, your investment request was declined by {name}.")
    return redirect(url_for('investor_dashboard_view', name=name))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
