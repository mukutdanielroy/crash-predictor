import time
import sqlite3
import asyncio
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Define constants
DB_PATH = 'prediction_data.db'
CHROME_DRIVER_PATH = r"Your chrome driver path"
URL = "https://1xbet.com/en/allgamesentrance/crash" #you need to change this according you your logged in URL

# Chrome options for headless mode
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# Initialize the Chrome WebDriver
driver = webdriver.Chrome(service=Service(CHROME_DRIVER_PATH), options=chrome_options)
driver.get(URL)

# Database operations
def init_db():
    """Initialize the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS crash_predictions (
            id INTEGER PRIMARY KEY,
            num_players INTEGER,
            total_bets FLOAT,
            actual_multiplier FLOAT,
            predicted_multiplier FLOAT
        )''')

def insert_prediction(num_players, total_bets, actual_multiplier, predicted_multiplier):
    """Insert prediction data into the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(''' 
        INSERT INTO crash_predictions (num_players, total_bets, actual_multiplier, predicted_multiplier)
        VALUES (?, ?, ?, ?)
        ''', (num_players, total_bets, actual_multiplier, predicted_multiplier))
        conn.commit()

def load_data():
    """Load data from the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        query = "SELECT num_players, total_bets, actual_multiplier FROM crash_predictions"
        data = pd.read_sql(query, conn)
    return data.dropna()

# Selenium operations
def wait_for_element(by, value, timeout=20):
    """Wait for an element to be present and return it."""
    return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((by, value)))

def get_timer():
    """Get the current timer value."""
    timer_element = wait_for_element(By.CLASS_NAME, "crash-timer__counter")
    return timer_element.text.strip()

def get_players():
    """Get the number of players."""
    players_element = wait_for_element(By.CLASS_NAME, "crash-total__value--players")
    return int(players_element.text.strip())

def get_bets():
    """Get the total bets value."""
    bets_element = wait_for_element(By.CLASS_NAME, "crash-total__value--bets")
    return float(bets_element.text.strip().replace("BDT", "").strip())

def get_current_value():
    """Get the current crash value."""
    time.sleep(0.5)
    current_value_element = wait_for_element(By.CLASS_NAME, "crash-game__counter")
    return round(float(current_value_element.text.strip().replace("x", "").strip()), 2)

# Data preprocessing
def create_lagged_features(data):
    """Create lagged features for the dataset."""
    data['previous_multiplier'] = data['actual_multiplier'].shift(1)
    data['players_change'] = data['num_players'].diff()
    data['bets_change'] = data['total_bets'].diff()
    return data.dropna()

def remove_outliers(data):
    """Remove outliers from the dataset using the IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Machine learning model training
def train_model():
    """Train the RandomForest model with hyperparameter tuning and cross-validation."""
    data = load_data()
    data = create_lagged_features(data)
    data = remove_outliers(data)

    X = data[['num_players', 'total_bets', 'previous_multiplier', 'players_change', 'bets_change']]
    y = data['actual_multiplier']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForest model
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    # Test the model and calculate metrics
    y_pred = rf_random.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Tuned Model MSE: {mse}, R-squared: {r2}")

    return rf_random.best_estimator_

# Prediction
def predict_multiplier(model, num_players, total_bets, previous_multiplier):
    """Predict the multiplier using the trained model."""
    input_data = pd.DataFrame([[num_players, total_bets, previous_multiplier, None, None]], 
                               columns=['num_players', 'total_bets', 'previous_multiplier', 'players_change', 'bets_change'])
    predicted_multiplier = model.predict(input_data)[0]
    return round(float(predicted_multiplier), 2)

# Asynchronous functions for data fetching
async def get_players_and_bets():
    """Fetch players and bets data after a delay."""
    await asyncio.sleep(9)  # Wait for the data to be available
    return get_players(), get_bets()

async def get_current_multiplier_after_delay():
    """Fetch current multiplier after a short delay."""
    await asyncio.sleep(0.6)
    return get_current_value()

# Main monitoring loop
async def monitor_game():
    """Monitor the game and perform predictions."""
    model = train_model()  # Initialize model outside the loop
    while True:
        try:
            timer = get_timer()

            if timer == "9":
                num_players, total_bets = await get_players_and_bets()
                previous_multiplier = 0.00

                predicted_multiplier = predict_multiplier(model, num_players, total_bets, previous_multiplier)
                print("################################################")
                print(f"Predicted Multiplier: {predicted_multiplier}")

                actual_multiplier = 0.00
                current_multiplier = 0.00
                while True:
                    try:
                        current_multiplier = await get_current_multiplier_after_delay()
                    except Exception as e:
                        actual_multiplier = current_multiplier
                        break

                if actual_multiplier is not None and actual_multiplier != 0.00:
                    print(actual_multiplier)
                    insert_prediction(num_players, total_bets, actual_multiplier, predicted_multiplier)
                    print(f"Stored data - Players: {num_players}, Bets: {total_bets}, Actual Multiplier: {actual_multiplier}, Predicted Multiplier: {predicted_multiplier}")
                    print("================================================")
                    # model = train_model()  # Optionally retrain the model with new data

        except Exception as e:
            print("An error occurred:", e)

# Initialize database and run the async function
init_db()

try:
    asyncio.run(monitor_game())
except Exception as e:
    print("An error occurred during monitoring:", e)
finally:
    driver.quit()  # Close the WebDriver
