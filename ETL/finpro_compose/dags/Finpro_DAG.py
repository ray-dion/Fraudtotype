'''
=========================================================
Final Project

This program is made to automate the process of loading
raw data from PostgreSQL, cleaning it, and posting it
to PostgreSQL.

=========================================================
'''

# Import Libraries
from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
from sqlalchemy import create_engine

# 1. FETCH DATA
def fetch():
    '''
    This function is used to fetch data from PostgreSQL. The data will be processed
    in the next function.
    '''
    # Create a SQLAlchemy engine for PostgreSQL
    engine = create_engine('postgresql+psycopg2://postgres:postgres@postgres:5432/finpro_hck_025')
    conn = engine.connect()

    # Find the designated data in postgres and save it in "df" variable
    df = pd.read_sql_query("SELECT * FROM table_finpro", conn)

    # Export the data in "df" to a csv file
    df.to_csv("/opt/airflow/data/Finpro_data_raw.csv", index=False)

    # Close connection
    conn.close()

# 2. CLEAN DATA
def clean():
    '''
    This function is used to process the previously fetched data.

    Processes:
    - Drop unnecessary columns, duplicates, and missing values
    - Convert data types (object to datetime)
    - Replace commas with dots and convert to numeric
    - Strip spaces and replace underscores with spaces in categorical columns
    - Delete data with 'fraud' string in the 'merchant' column
    - Fix capitalization in "category" column
    - Feature Creation: 'Age' and 'Hour'

    Output: Cleaned data in CSV format

    '''
    # Read the previously fetched data
    df = pd.read_csv("/opt/airflow/data/Finpro_data_raw.csv")

    # Drop unnecessary columns
    df.drop(columns=['unnamed_0', 'unnamed_01'], inplace=True)

    # Drop duplicates and missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True) 

    # Convert data type to datetime
    date_cols = ["trans_date_trans_time", "dob", "unix_time"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Replace commas with dots and convert to numeric
    comma_to_float_cols = ["amt", "lat", "long", "merch_lat", "merch_long"]
    for col in comma_to_float_cols:
        df[col] = df[col].str.replace(",", ".").astype(float)

    # Strip spaces and and replace underscores with spaces in categorical columns
    cat_cols = ["merchant", "category", "first", "last", "gender", "street", "city", "state", "job"]
    for col in cat_cols:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace("_", " ")

    # Delete data with 'fraud' string in the 'merchant' column
    df['merchant'] = df['merchant'].str.replace("fraud", "")

    # Fix capitalization in "category" column
    df['category'] = df['category'].str.title()

    # Create a new column 'trans_hour'
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour

    # Create a new column 'age'
    latest_date = df['trans_date_trans_time'].max()
    df['age'] = (latest_date - df['dob']).dt.days // 365

    # Export the processed data to a new csv file
    df.to_csv("/opt/airflow/data/Finpro_data_clean.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

# 3. POST TO POSTGRESQL
def post():
    '''
    This function is used to post the processed data to PostgreSQL.
    '''

    # Create a SQLAlchemy engine for PostgreSQL
    engine = create_engine('postgresql+psycopg2://postgres:postgres@postgres:5432/finpro_hck_025')
    conn = engine.connect()

    # Read the data we want to send
    df = pd.read_csv("/opt/airflow/data/Finpro_data_clean.csv", parse_dates=["trans_date_trans_time", "dob", "unix_time"])

    # Send the data to PostgreSQL
    df.to_sql('table_finpro_cleaned', conn, if_exists='replace', index=False)

    # Close connection
    conn.close()

# Define DAG
default_args = {
    'owner': 'thaliban',
    'start_date': datetime(2025, 4, 21),
    'retries': 1
}

with DAG(
    # Define initial parameters
    dag_id='finpro_data_pipeline',
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    description="Fetch -> Clean -> Post") as dag:

    # Define operations within the DAG
    # Fetch Data
    fetch_data = PythonOperator(
        task_id='fetch_from_postgresql',
        python_callable=fetch
    )
    # Clean Data
    clean_data = PythonOperator(
        task_id='data_cleaning',
        python_callable=clean
    )
    # Post Data
    post_data = PythonOperator(
        task_id='post_to_postgresql',
        python_callable=post
    )

    # DAG Flow
    fetch_data >> clean_data >> post_data
