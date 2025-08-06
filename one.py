import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objs as plt
import mysql.connector
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import os
from openpyxl import Workbook

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd

from prediction import SarimaXGBoostEnsembleModel


# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


API_KEY = "579b464db66ec23bdd000001c4f6e3a1106c4281715a8bd45f4197ed"
WEATHER_API_KEY = "f334e2ba66ffa5970c4207243b7b7494"

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Password',
    'database': 'database1'
}

class CPIFetcher:
    def __init__(self, cpi_file_path: str):
        self.cpi_file_path = cpi_file_path
        self.cpi_data = None

    def load_cpi_data(self):
        """Load CPI data from CSV file and preprocess it."""
        try:
            self.cpi_data = pd.read_csv(self.cpi_file_path)

            self.cpi_data['Arrival_Date'] = pd.to_datetime(self.cpi_data['Arrival_Date'], format='%d-%m-%Y').dt.strftime('%m-%d-%Y')                 # Convert date from dd-mm-yyyy to mm-dd-yyyy
            self.cpi_data['Arrival_Date'] = pd.to_datetime(self.cpi_data['Arrival_Date'], format='%m-%d-%Y')

            self.cpi_data['State'] = self.cpi_data['State'].str.strip()  # Standardize state names
            logger.info("CPI data loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading CPI data: {e}")



def display_commodity_data(df_prices, price_columns):
    """Display commodity data with summary statistics and trends."""
    df_prices[price_columns] = df_prices[price_columns].apply(pd.to_numeric, errors='coerce')

 
    st.subheader("📊 Price Statistics")
    summary_stats = df_prices[price_columns].describe()
    st.dataframe(summary_stats.style.format("{:.2f}"))

    st.write("### Modal Price vs CPI")
    fig1 = plt.Figure(data=[
        plt.Scatter(x=df_prices['CPI'], y=df_prices['Modal_Price'], mode='markers', name='CPI vs Modal Price')
    ])
    fig1.update_layout(
        title='Modal Price vs CPI',
        xaxis_title='CPI',
        yaxis_title='Modal Price',
        hovermode='closest'
    )
    st.plotly_chart(fig1)

 
    st.write("### Modal Price vs Temperature")
    fig2 = plt.Figure(data=[
        plt.Scatter(x=df_prices['temperature'], y=df_prices['Modal_Price'], mode='markers', name='Temperature vs Modal Price')
    ])
    fig2.update_layout(
        title='Modal Price vs Temperature',
        xaxis_title='Temperature',
        yaxis_title='Modal Price',
        hovermode='closest'
    )
    st.plotly_chart(fig2)


    st.write("### Impact of CPI and Temperature on Modal Price")
    cpi_correlation = df_prices['CPI'].corr(df_prices['Modal_Price'])
    temp_correlation = df_prices['temperature'].corr(df_prices['Modal_Price'])
    
    if cpi_correlation > 0.5:
        st.write("- Higher CPI values strongly correlate with higher modal prices, indicating inflationary effects.")
    elif cpi_correlation > 0:
        st.write("- CPI has a slight positive impact on modal prices, but other factors may influence price fluctuations.")
    else:
        st.write("- CPI does not significantly impact modal prices in this dataset.")
    
    if temp_correlation > 0.5:
        st.write("- Higher temperatures are associated with increased modal prices, possibly due to supply chain issues or spoilage risks.")
    elif temp_correlation > 0:
        st.write("- Temperature has a minor effect on modal prices, indicating seasonality trends.")
    else:
        st.write("- Temperature does not significantly impact modal prices in this dataset.")


    st.write("### Raw Data Table:")
    page_size = 10
    total_pages = len(df_prices) // page_size + (1 if len(df_prices) % page_size > 0 else 0)
    
    col1, col2 = st.columns([7, 3])
    with col2:
        page_number = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)

    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size

    display_df = df_prices.iloc[start_idx:end_idx].copy()
    display_df['Arrival_Date'] = display_df['Arrival_Date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(display_df)


    csv = df_prices.to_csv(index=False)
    st.download_button("Download Data as CSV", data=csv, file_name="commodity_data.csv", mime="text/csv")



def process_and_merge_data(price_records: List[Dict], weather_data: List[Dict], cpi_fetcher: CPIFetcher) -> Optional[pd.DataFrame]:
    """Merge commodity, CPI, and weather data based on Date and State/District."""
    if not price_records or not weather_data:
        logger.warning("No data available for processing")
        return None

    df_prices = pd.DataFrame(price_records)
    df_weather = pd.DataFrame(weather_data)

    # Convert dates
    df_prices['Arrival_Date'] = pd.to_datetime(df_prices['Arrival_Date'], errors='coerce')
    df_weather['fetch_date'] = pd.to_datetime(df_weather['fetch_date'], errors='coerce')
    df_prices['State'] = df_prices['State'].str.strip()
    df_prices['District'] = df_prices['District'].str.strip()

    # Merge commodity and CPI data using 'State' and 'Arrival_Date'
    df_prices = df_prices.merge(cpi_fetcher.cpi_data[['State', 'Arrival_Date', 'Consumer_Price_Index']], on=['State', 'Arrival_Date'], how='left')

    # Ensure the CPI column exists and rename correctly
    df_prices.rename(columns={'Consumer_Price_Index': 'CPI'}, inplace=True)
    
    # Merge with weather data using 'fetch_date' (matching Arrival_Date)
    df_combined = pd.merge(df_prices, df_weather[['fetch_date', 'temperature', 'visibility', 'wind_speed', 'clouds']], left_on=['Arrival_Date'], right_on=['fetch_date'], how='left')
    df_combined.drop(columns=['fetch_date'], inplace=True)

    # Select only required columns
    df_combined = df_combined[['State', 'District', 'Market', 'Commodity', 'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price', 'temperature', 'visibility', 'wind_speed', 'clouds', 'CPI']]

    # Fill missing values using new approach
    numeric_columns = df_combined.select_dtypes(include=['float64', 'int64']).columns
    df_combined[numeric_columns] = df_combined[numeric_columns].ffill().bfill()

    df_combined['processing_timestamp'] = pd.Timestamp.now()
    df_combined['data_version'] = '1.0'

    logger.info(f"Processed {len(df_combined)} records")
    return df_combined



class DatabaseHandler:
    def __init__(self, host: str = DB_CONFIG['host'], user: str = DB_CONFIG['user'], 
                 password: str = DB_CONFIG['password'], database: str = DB_CONFIG['database']):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                self._create_tables()
        except Exception as e:
            st.error(f"Database error: {e}")
            raise

    def _create_tables(self):
        try:
            cursor = self.connection.cursor()
            
            tables = {
                'commodity_data': """
                CREATE TABLE IF NOT EXISTS commodity_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    commodity VARCHAR(100),
                    state VARCHAR(100),
                    district VARCHAR(100),
                    arrival_date DATE,
                    min_price FLOAT,
                    max_price FLOAT,
                    modal_price FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB;
                """,
                'weather_data': """
                CREATE TABLE IF NOT EXISTS weather_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    city VARCHAR(100),
                    temperature FLOAT,
                    visibility INT,
                    wind_speed FLOAT,
                    clouds INT,
                    country VARCHAR(10),
                    fetch_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB;
                """,
                'commodity_weather_data': """
                CREATE TABLE IF NOT EXISTS commodity_weather_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    commodity VARCHAR(100),
                    state VARCHAR(100),
                    district VARCHAR(100),
                    min_price FLOAT,
                    max_price FLOAT,
                    modal_price FLOAT,
                    temperature FLOAT,
                    visibility INT,
                    wind_speed FLOAT,
                    clouds INT,
                    city VARCHAR(100),
                    country VARCHAR(10),
                    record_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_record_date (record_date),
                    INDEX idx_commodity (commodity),
                    INDEX idx_location (state, district)
                ) ENGINE=InnoDB;
                """
            }
            
            for query in tables.values():
                cursor.execute(query)
                
            self.connection.commit()
            cursor.close()
            logger.info("Successfully created/verified all required database tables")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise

    def get_unique_commodities(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT DISTINCT commodity FROM commodity_data")
            commodities = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return commodities
        except Exception as e:
            st.error(f"Database error: {e}")
            return []

    def get_states_for_commodity(self, commodity):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT DISTINCT state FROM commodity_data WHERE commodity = %s", 
                (commodity,)
            )
            states = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return states
        except Exception as e:
            st.error(f"Database error: {e}")
            return []

    def get_districts_for_state_commodity(self, state, commodity):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT DISTINCT district FROM commodity_data WHERE state = %s AND commodity = %s", 
                (state, commodity)
            )
            districts = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return districts
        except Exception as e:
            st.error(f"Database error: {e}")
            return []

    def save_commodity_data(self, data: List[Dict]):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
                
            cursor = self.connection.cursor()
            
            insert_query = """
            INSERT INTO commodity_data 
            (commodity, state, district, arrival_date, min_price, max_price, modal_price)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            for record in data:
                arrival_date_str = record.get('Arrival_Date', '')
                try:
                    arrival_date = datetime.strptime(arrival_date_str, '%d/%m/%Y')
                except ValueError:
                    try:
                        arrival_date = datetime.strptime(arrival_date_str, '%d-%m-%Y')
                    except ValueError:
                        logger.error(f"Unable to parse date: {arrival_date_str}")
                        continue

                values = (
                    record.get('Commodity'),
                    record.get('State'),
                    record.get('District'),
                    arrival_date,
                    float(record.get('Min_Price', 0)),
                    float(record.get('Max_Price', 0)),
                    float(record.get('Modal_Price', 0))
                )
                cursor.execute(insert_query, values)
                
            self.connection.commit()
            cursor.close()
            logger.info(f"Successfully saved {len(data)} commodity records to database")
            
        except Exception as e:
            logger.error(f"Error saving commodity data to database: {str(e)}")
            if self.connection:
                self.connection.rollback()
            raise

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")

    def get_unique_commodities(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT DISTINCT commodity FROM commodity_data")
            commodities = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
         
            if not commodities:
                default_commodities = [
                    "Wheat", "Rice", "Tur(Arhar)Dal", "Potato", "Onion", 
                    "Tomato", "Soyabean", "Moong Dal", "Sugar","Gur","Groundnut Oil",
                    "Mustard Oil","Tea","Bajra","Masur Dal"
                ]
                return default_commodities
            return commodities
        except Exception as e:
            st.error(f"Database error: {e}")
            return ["Wheat"] 

    def get_states_for_commodity(self, commodity):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT DISTINCT state FROM commodity_data WHERE commodity = %s", 
                (commodity,)
            )
            states = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
       
            if not states:
                default_states = [
                    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", 
                    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", 
                    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", 
                    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", 
                    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", 
                    "Uttar Pradesh", "Uttarakhand", "West Bengal"
                ]

                return default_states
            return states
        except Exception as e:
            st.error(f"Database error: {e}")
            return ["Maharashtra"] 

    def get_districts_for_state_commodity(self, state, commodity):
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT DISTINCT district FROM commodity_data WHERE state = %s AND commodity = %s", 
                (state, commodity)
            )
            districts = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
     
            if not districts:
                default_districts = {
                    "Andhra Pradesh": ["Anantapur", "Chittoor", "East Godavari", "Guntur", "Krishna", "Kurnool", "Prakasam", "Srikakulam", "Visakhapatnam", "West Godavari", "YSR Kadapa", "Vijayawada", "Tirupati", "Kakinada", "Guntur", "Vijayanagaram"],
                    "Arunachal Pradesh": ["Tawang", "West Kameng", "East Kameng", "Papum Pare", "Kurung Kumey", "Kra Daadi", "Lower Subansiri", "Upper Subansiri", "West Siang", "East Siang", "Siang", "Lower Siang", "Changlang", "Longding", "Lohit", "Anjaw", "Namsai"],
                    "Assam": ["Baksa", "Barpeta", "Bongaigaon", "Cachar", "Charaideo", "Darrang", "Dhemaji", "Dibrugarh", "Goalpara", "Golaghat", "Hailakandi", "Jorhat", "Kamrup", "Kamrup Metro", "Karbi Anglong", "Karimganj", "Kokrajhar", "Lakhimpur", "Majuli", "Morigaon", "Nagaon", "Nalbari", "Sivasagar", "Sonitpur", "Tinsukia", "Udalguri", "West Karbi Anglong"],
                    "Bihar": ["Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur", "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj", "Jamui", "Jehanabad", "Kaimur", "Katihar", "Khagaria", "Kishanganj", "Lakhisarai", "Madhubani", "Munger", "Muzaffarpur", "Nalanda", "Nawada", "Patna", "Purnia", "Rohtas", "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Supaul", "Vaishali", "West Champaran"],
                    "Chhattisgarh": ["Balod", "Baloda Bazar", "Balrampur", "Bemetara", "Bijapur", "Bilaspur", "Dantewada", "Dhamtari", "Durg", "Gariaband", "Janjgir-Champa", "Jashpur", "Kabirdham", "Kanker", "Korba", "Koriya", "Mahasamund", "Mungeli", "Narayanpur", "Raigarh", "Raipur", "Rajnandgaon", "Surajpur", "Surguja"],
                    "Goa": ["North Goa", "South Goa"],
                    "Gujarat": ["Ahmedabad", "Amreli", "Anand", "Aravalli", "Banaskantha", "Bharuch", "Bhavnagar", "Botad", "Chhota Udaipur", "Dahod", "Dangs", "Gandhinagar", "Gir Somnath", "Jamnagar", "Junagadh", "Kachchh", "Kheda", "Mahisagar", "Mehsana", "Morbi", "Narmada", "Navsari", "Panchmahal", "Patan", "Porbandar", "Rajkot", "Sabarkantha", "Surat", "Surendranagar", "Tapi", "Vadodara", "Valsad"],
                    "Haryana": ["Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad", "Gurugram", "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal", "Kurukshetra", "Mahendragarh", "Nuh", "Palwal", "Panchkula", "Panipat", "Rewari", "Sirsa", "Sonipat", "Yamunanagar"],
                    "Himachal Pradesh": ["Bilaspur", "Chamba", "Hamirpur", "Kangra", "Kullu", "Mandi", "Shimla", "Sirmaur", "Solan", "Una"],
                    "Jharkhand": ["Bokaro", "Chatra", "Deoghar", "Dhanbad", "Dumka", "Giridih", "Jamshedpur", "Khunti", "Koderma", "Latehar", "Lohardaga", "Pakur", "Palamu", "Ranchi", "Sahibganj", "Saraikela Kharsawan", "Simdega", "West Singhbhum"],
                    "Karnataka": ["Bagalkot", "Bangalore Rural", "Bangalore Urban", "Belgaum", "Bellary", "Bidar", "Chamarajanagar", "Chikkamagaluru", "Chikkaballapur", "Chitradurga", "Dakshina Kannada", "Davangere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kodagu", "Kolar", "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumkur", "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir"],
                    "Kerala": ["Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod", "Kottayam", "Kollam", "Kozhikode", "Malappuram", "Palakkad", "Pathanamthitta", "Thiruvananthapuram", "Thrissur", "Wayanad"],
                    "Madhya Pradesh": ["Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani", "Betul", "Bhind", "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh", "Datia", "Dewas", "Dhar", "Dindori", "Guna", "Gwalior", "Harda", "Hoshangabad", "Indore", "Jabalpur", "Jhabua", "Katni", "Khandwa", "Khargone", "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch", "Panna", "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", "Seoni", "Shahdol", "Shajapur", "Sheopur", "Shivpuri", "Singrauli", "Tikamgarh", "Ujjain", "Umaria", "Vidisha"],
                    "Maharashtra": ["Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai", "Mumbai Suburban", "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sakoli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"],
                    "Manipur": ["Bishnupur", "Chandel", "Churachandpur", "Imphal East", "Imphal West", "Jiribam", "Kangpokpi", "Noney", "Senapati", "Tamenglong", "Tengnoupal", "Thoubal", "Ukhrul"],
                    "Meghalaya": ["East Garo Hills", "East Khasi Hills", "Jaintia Hills", "Ri Bhoi", "South Garo Hills", "South West Garo Hills", "South West Khasi Hills", "West Garo Hills", "West Khasi Hills"],
                    "Mizoram": ["Aizawl", "Champhai", "Kolasib", "Lunglei", "Mamit", "Serchhip"],
                    "Nagaland": ["Dimapur", "Kohima", "Mokokchung", "Mon", "Peren", "Phek", "Tuensang", "Wokha", "Zunheboto"],
                    "Odisha": ["Angul", "Bargarh", "Bhadrak", "Balasore", "Cuttack", "Dhenkanal", "Ganjam", "Gajapati", "Jagatsinghpur", "Jajpur", "Jharsuguda", "Kalahandi", "Kandhamal", "Kendrapara", "Keonjhar", "Khurda", "Koraput", "Malkangiri", "Mayurbhanj", "Nabarangpur", "Nayagarh", "Nuapada", "Puri", "Rayagada", "Sambalpur", "Sundargarh"],
                    "Punjab": ["Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib", "Firozpur", "Gurdaspur", "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana", "Mansa", "Moga", "Muktsar", "Nawanshahr", "Patiala", "Rupnagar", "Sangrur", "SBS Nagar", "Shahid Bhagat Singh Nagar", "Tarn Taran"],
                    "Rajasthan": ["Ajmer", "Alwar", "Banswara", "Baran", "Barmer", "Bhilwara", "Bikaner", "Bundi", "Chittorgarh", "Churu", "Dausa", "Dholpur", "Dungarpur", "Hanumangarh", "Jaisalmer", "Jaipur", "Jalore", "Jhalawar", "Jhunjhunu", "Jodhpur", "Karauli", "Kota", "Nagaur", "Pali", "Rajsamand", "Sikar", "Sirohi", "Sri Ganganagar", "Tonk", "Udaipur"],
                    "Sikkim": ["East Sikkim", "North Sikkim", "South Sikkim", "West Sikkim"],
                    "Tamil Nadu": ["Chennai", "Coimbatore", "Cuddalore", "Dharmapuri", "Dindigul", "Erode", "Kancheepuram", "Kanyakumari", "Karur", "Krishnagiri", "Madurai", "Nagapattinam", "Namakkal", "Perambalur", "Pudukkottai", "Ramanathapuram", "Salem", "Sivaganga", "Tenkasi", "Thanjavur", "The Nilgiris", "Theni", "Tiruvallur", "Tirunelveli", "Tirupur", "Tiruvannamalai", "Vellore", "Villupuram", "Virudhunagar"],
                    "Telangana": ["Adilabad", "Hyderabad", "Jagitial", "Jangaon", "Khammam", "Karimnagar", "Khammam", "Mahabubnagar", "Medak", "Nalgonda", "Nirmal", "Nizamabad", "Peddapalli", "Rangareddy", "Sangareddy", "Warangal", "Khammam"],
                    "Tripura": ["Dhalai", "Khowai", "North Tripura", "South Tripura", "West Tripura"],
                    "Uttar Pradesh": ["Agra", "Aligarh", "Ambedkar Nagar", "Amethi", "Amroha", "Auraiya", "Ayodhya", "Azamgarh", "Baghpat", "Bahraich", "Ballia", "Balrampur", "Banda", "Barabanki", "Bareilly", "Basti", "Bijnor", "Budaun", "Bulandshahr", "Chandauli", "Chitrakoot", "Deoria", "Etah", "Etawah", "Faizabad", "Farrukhabad", "Fatehpur", "Firozabad", "Gautam Buddha Nagar", "Ghaziabad", "Gorakhpur", "Hamirpur", "Hapur", "Hardoi", "Hathras", "Jhansi", "Kannauj", "Kanpur", "Kaushambi", "Kushinagar", "Lakhimpur Kheri", "Lucknow", "Maharajganj", "Mainpuri", "Mathura", "Meerut", "Mirzapur", "Moradabad", "Muzaffarnagar", "Pratapgarh", "Raebareli", "Rampur", "Saharanpur", "Shahjahanpur", "Shamli", "Shravasti", "Siddharthnagar", "Sonbhadra", "Sultanpur", "Unnao", "Varanasi"],
                    "Uttarakhand": ["Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar", "Nainital", "Pauri Garhwal", "Pithoragarh", "Rudraprayag", "Tehri Garhwal", "Udham Singh Nagar", "Uttarkashi"],
                    "West Bengal": ["Alipurduar", "Bankura", "Birbhum", "Cooch Behar", "Dakshin Dinajpur", "Hooghly", "Howrah", "Jalpaiguri", "Jhargram", "Kalimpong", "Kolkata", "Malda", "Murshidabad", "Nadia", "North 24 Parganas", "Paschim Medinipur", "Purba Medinipur", "Purulia", "South 24 Parganas", "Uttar Dinajpur"]
                }

                return default_districts.get(state, ["Nashik"]) 
            return districts
        except Exception as e:
            st.error(f"Database error: {e}")
            return ["Nashik"] 



class WeatherDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def fetch_weather_data(self, city, date):
        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            return {
                'city': city,
                'temperature': data['main']['temp'],
                'visibility': data.get('visibility', 0),
                'wind_speed': data['wind']['speed'],
                'clouds': data.get('clouds', {}).get('all', 0),
                'country': data['sys']['country'],
                'fetch_date': date
            }
        except Exception as e:
            st.error(f"Weather data fetch error: {e}")
            return None

    def fetch_weather_data_in_range(self, city: str, start_date: str, end_date: str) -> List[Dict]:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Date parsing error: {str(e)}")
            return []

        weather_data = []
        current_date = start

        while current_date <= end:
            data = self.fetch_weather_data(city, current_date.strftime('%Y-%m-%d'))
            if data:
                weather_data.append(data)
            current_date += timedelta(days=1)

        return weather_data



class CommodityDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"

    def fetch_data_in_range(self, state: str, district: str, commodity: str, 
                           start_date: str, end_date: str) -> List[Dict]:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            st.error(f"Date parsing error: {str(e)}")
            return []

        all_records = []
        current_date = start

        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%d-%m-%Y')
            except ValueError:
                st.error(f"Date conversion error for {date_str}")
                continue

            params = {
                "api-key": self.api_key,
                "format": "json",
                "filters[State.keyword]": state,
                "filters[District.keyword]": district,
                "filters[Commodity.keyword]": commodity,
                "filters[Arrival_Date]": formatted_date,
                "offset": 0,
                "limit": 1000
            }

            try:
                response = requests.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                daily_records = data.get("records", [])
                all_records.extend(daily_records)

                st.info(f"Fetched {len(daily_records)} records for {date_str}")
                current_date += timedelta(days=1)
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed for date {date_str}: {str(e)}")

        return all_records



class ExcelHandler:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def save_to_excel(self, df: pd.DataFrame, filename: str):
        try:
            filepath = os.path.join(self.output_dir, filename)
            with pd.ExcelWriter(filepath, engine='openpyxl', datetime_format='YYYY-MM-DD') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            logger.info(f"Successfully saved data to Excel file: {filepath}")
        except Exception as e:
            logger.error(f"Error saving to Excel: {str(e)}")
            raise




st.set_page_config(
    page_title="KRUSH!",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)




st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {
        padding: 1rem 1rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)



def render_home_page():
    st.title("🌾 Welcome to KRUSH!")
    

    st.markdown("""
    <div style='padding: 2em; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 2em;'>
        <h3 style='color: #2c3e50; text-align: center;'>
            "Agriculture is our wisest pursuit, because it will in the end contribute most to real wealth, good morals & happiness."
        </h3>
        <p style='text-align: right; color: #7f8c8d;'>- Thomas Jefferson</p>
    </div>
    """, unsafe_allow_html=True)



def render_price_analysis():
    st.title('Commodity Price Analysis')

    try:
        db_handler = DatabaseHandler()
        db_handler.connect()

        cpi_fetcher = CPIFetcher("Final_CPI_Daily_2011_2023.csv")
        cpi_fetcher.load_cpi_data()

        with st.sidebar:
            st.header("Data Selection")
            commodities = db_handler.get_unique_commodities()
            commodity = st.selectbox('Select Commodity', commodities)

            states = db_handler.get_states_for_commodity(commodity)
            state = st.selectbox('Select State', states)

            districts = db_handler.get_districts_for_state_commodity(state, commodity)
            district = st.selectbox('Select District', districts)

            min_date = datetime.now() - timedelta(days=15*365)
            max_date = datetime.now()
            
            start_date = st.date_input('Start Date', value=datetime.now() - timedelta(days=30), min_value=min_date, max_value=max_date)
            end_date = st.date_input('End Date', value=datetime.now(), min_value=start_date, max_value=max_date)

            if start_date > end_date:
                st.error("Start date must be before end date")
                return

            fetch_data = st.button('Fetch Data')

        if fetch_data:
            try:
                commodity_fetcher = CommodityDataFetcher(API_KEY)
                weather_fetcher = WeatherDataFetcher(WEATHER_API_KEY)
                excel_handler = ExcelHandler()

                commodity_records = commodity_fetcher.fetch_data_in_range(state, district, commodity, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if not commodity_records:
                    st.warning("No commodity data available for the selected period")
                    return

                weather_records = weather_fetcher.fetch_weather_data_in_range(city=district, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
                if not weather_records:
                    st.warning("No weather data available for the selected period")
                    return

                df_combined = process_and_merge_data(commodity_records, weather_records, cpi_fetcher)

                if df_combined is None or df_combined.empty:
                    st.error("Failed to process data. Please try different parameters.")
                    return

                excel_handler.save_to_excel(df_combined, "combined_data.xlsx")
                st.success("Data saved to Excel successfully!")

                display_commodity_data(df_combined, ['Min_Price', 'Max_Price', 'Modal_Price'])

            except Exception as e:
                st.error(f"Error during data processing: {str(e)}")
                logger.error(f"Data processing error: {str(e)}")

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
    
    finally:
        try:
            db_handler.close()
        except:
            pass

def render_price_prediction():
    st.title("Commodity Price Prediction")
    
    try:
        db_handler = DatabaseHandler()
        db_handler.connect()

        col1, col2 = st.columns(2)
        with col1:
            commodities = db_handler.get_unique_commodities()
            commodity = st.selectbox('Select Commodity', commodities)
            states = db_handler.get_states_for_commodity(commodity)
            state = st.selectbox('Select State', states)

        with col2:
            districts = db_handler.get_districts_for_state_commodity(state, commodity)
            district = st.selectbox('Select District', districts)
            prediction_date = st.date_input(
                'Select Date for Prediction',
                value=datetime.now(),
                min_value=datetime.now() - timedelta(days=365 * 25),
                max_value=datetime.now() + timedelta(days=365)
            )

        # Set confidence level directly in code - fixed at 95%
        confidence_level = 95

        perform_backtesting = st.checkbox("Perform Backtesting", value=True)

        if st.button('Generate Prediction'):
            with st.spinner('Processing data and generating predictions...'):
                try:
                    # Fetch historical data
                    commodity_fetcher = CommodityDataFetcher(API_KEY)
                    start_date = prediction_date - timedelta(days=365)
                    historical_data = commodity_fetcher.fetch_data_in_range(
                        state, district, commodity,
                        start_date.strftime('%Y-%m-%d'),
                        prediction_date.strftime('%Y-%m-%d')
                    )

                    if not historical_data:
                        st.warning("No historical data available for prediction")
                        return

                    # Process data
                    df = pd.DataFrame(historical_data)
                    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='mixed', dayfirst=True)
                    df = df.sort_values('Arrival_Date')

                    # Initialize and train model
                    model = SarimaXGBoostEnsembleModel()
                    with st.spinner('Training model... This may take a minute.'):
                        success = model.train(df)
                        if not success:
                            st.error("Failed to train model. Please try another commodity or location.")
                            return

                    # Perform backtesting if selected
                    if perform_backtesting:
                        with st.spinner('Performing backtesting...'):
                            results = model.backtesting(df)
                            if results:
                                st.subheader("Backtesting Results")
                                
                                # Display metrics
                                metrics = results['metrics']
                                cols = st.columns(4)
                                cols[0].metric("RMSE", f"₹{metrics['rmse']:.2f}")
                                cols[1].metric("MAE", f"₹{metrics['mae']:.2f}")
                                cols[2].metric("R² Score", f"{metrics['r2']:.3f}")
                                cols[3].metric("MAPE", f"{metrics['mape']:.2f}%")
                                
                                # Plot results
                                st.subheader("Backtesting Visualization")
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=results['test_dates'],
                                    y=results['actual_prices'],
                                    name='Actual Prices',
                                    line=dict(color='blue', width=2)
                                ))
                                fig.add_trace(go.Scatter(
                                    x=results['test_dates'],
                                    y=results['ensemble_predictions'],
                                    name='Predicted Prices',
                                    line=dict(color='red', width=2)
                                ))
                                
                                fig.update_layout(
                                    title='Model Performance on Test Data',
                                    xaxis_title='Date',
                                    yaxis_title='Price (₹)',
                                    hovermode='x unified',
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    # Generate future predictions
                    with st.spinner('Generating price predictions...'):
                        predictions = model.predict(prediction_date.strftime('%Y-%m-%d'))
                        
                        if predictions.empty:
                            st.error("Failed to generate predictions. Please try another date or commodity.")
                            return
                        
                        # Continue only if predictions are not empty
                        st.subheader("Price Predictions for Next 5 Days")
                        
                        # Debug: Display dataframe column names and head
                        st.write(f"Debug: Prediction columns: {predictions.columns.tolist()}")
                        
                        # Ensure predictions have proper columns
                        required_columns = ['date', 'predicted_price', 'lower_bound', 'upper_bound']
                        missing_columns = [col for col in required_columns if col not in predictions.columns]
                        
                        if missing_columns:
                            st.error(f"Missing required columns in predictions: {missing_columns}")
                            st.write("Available columns:", predictions.columns.tolist())
                            st.write("Preview of predictions dataframe:", predictions.head())
                            return
                        
                        # Display predicted prices for the next 5 days
                        st.markdown(f"### Predicted Prices for {commodity} in {district}, {state}")
                        
                        # Create a table to display the predictions
                        price_data = {
                            "Date": [date.strftime('%d %b %Y') for date in predictions['date']],
                            "Predicted Price (₹)": [f"₹{price:.2f}" for price in predictions['predicted_price']],
                            f"Price Range ({confidence_level}% CI)": [
                                f"₹{lower:.2f} - ₹{upper:.2f}" 
                                for lower, upper in zip(predictions['lower_bound'], predictions['upper_bound'])
                            ]
                        }
                        
                        # Display the prediction table
                        st.table(pd.DataFrame(price_data))
                        
                        # Create prediction cards with ranges
                        st.markdown("### Prediction Cards")
                        cols = st.columns(5)
                        for idx, (_, row) in enumerate(predictions.iterrows()):
                            with cols[idx]:
                                st.markdown(f"""
                                <div style='padding: 1em; background-color: white; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center;'>
                                    <p style='color: #666; font-size: 0.9em;'>{row['date'].strftime('%d %b %Y')}</p>
                                    <h3 style='color: #2c3e50; margin: 0;'>₹{row['predicted_price']:.2f}</h3>
                                    <p style='color: #666; font-size: 0.8em;'>Range: ₹{row['lower_bound']:.2f} - ₹{row['upper_bound']:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Plot predictions with confidence intervals
                        fig = go.Figure()
                        
                        # Add predicted price line
                        fig.add_trace(go.Scatter(
                            x=predictions['date'],
                            y=predictions['predicted_price'],
                            name='Price Prediction',
                            line=dict(color='red', width=2)
                        ))
                        
                        # Add confidence interval as a filled area
                        fig.add_trace(go.Scatter(
                            x=predictions['date'].tolist() + predictions['date'].tolist()[::-1],
                            y=predictions['upper_bound'].tolist() + predictions['lower_bound'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(231,107,107,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=True,
                            name=f'{confidence_level}% Confidence Interval'
                        ))
                        
                        fig.update_layout(
                            title=f'5-Day Price Forecast with {confidence_level}% Confidence Interval',
                            xaxis_title='Date',
                            yaxis_title='Price (₹)',
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Application error: {str(e)}")
    finally:
        try:
            db_handler.close()
        except:
            pass

def main():

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Price Analysis", "Price Prediction"],
        key="navigation"
    )
    

    st.empty()
    

    if page == "Home":
        render_home_page()
    elif page == "Price Analysis":
        render_price_analysis()
    else:
        render_price_prediction()



if __name__ == "__main__":
    main()




































