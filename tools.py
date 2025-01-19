import os
from dotenv import load_dotenv

import pandas as pd
from sqlalchemy import create_engine
import psycopg2
import csv

load_dotenv()
os.environ['POSTGRES_DB_URL'] = os.getenv("POSTGRES_DB_URL")

# db_url = 'postgresql://postgres:admin@localhost:5432/olist'
db_url = os.getenv("POSTGRES_DB_URL")

engine = create_engine(db_url)


# def analyze_csv_with_llm(csv_path: str):
#     try:
#         # Read CSV content
#         with open(csv_path, "r") as csvfile:
#             reader = csv.reader(csvfile)
#             rows = list(reader)
        
#         # Prepare a prompt for the LLM
#         header = ", ".join(rows[0])
#         data_sample = "\n".join([", ".join(row) for row in rows[1:6]])  # Use a sample of first 5 rows
#         prompt = (
#             f"Here is a dataset with the following columns: {header}.\n"
#             f"Sample data:\n{data_sample}\n\n"
#             "Analyze this dataset and provide insights, trends, and actionable recommendations."
#         )
        
#         # Generate insights
#         insights = llm(prompt)
#         return insights
#     except Exception as e:
#         return f"An error occurred during analysis: {str(e)}"
    


# def execute_query_tool_tocsv(sql_query):
#     try:
#         # Connect to your postgres DB

#         formatted_sql_query = f"""{sql_query}"""
#         conn = psycopg2.connect(db_url)
#         # Open a cursor to perform database operations
#         cur = conn.cursor()

#         # Execute the SQL query
#         cur.execute(formatted_sql_query)

#         # Fetch the result
#         result = cur.fetchall()

#         print(result)
#         columns = [desc[0] for desc in cur.description]
        
#         # Save results to a CSV file
#         # csv_path = os.path.join(CSV_OUTPUT_DIR, 'csv_results.csv')
#         with open(CSV_FILE_PATH, "w", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(columns)  # Write headers
#             writer.writerows(result)  # Write rows
            
#         # Close communication with the database
#             cur.close()
#             conn.close()
#         return True
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None


def execute_query_tool(sql_query):
    try:
        # Connect to your postgres DB
        conn = psycopg2.connect(db_url)
        # Open a cursor to perform database operations
        cur = conn.cursor()

        # Format the SQL query
        formatted_sql_query = f"""{sql_query}"""

        # Execute the SQL query
        cur.execute(formatted_sql_query)

        # Fetch the result
        result = cur.fetchall()

        # print('resulst from code --', result)
        columns = [desc[0] for desc in cur.description]
        
        # Convert the result to a DataFrame
        df = pd.DataFrame(result, columns=columns)
        
        # Close communication with the database
        cur.close()
        conn.close()

        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
