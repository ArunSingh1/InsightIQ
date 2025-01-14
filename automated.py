import csv
import os
from sqlalchemy import create_engine, text
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI

# Database connection details
DB_NAME = "olist"
DB_USERNAME = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"

# db_url = f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
db_url = 'postgresql://postgres:admin@localhost:5432/olist'
engine = create_engine(db_url)

# Directory to store CSV files
CSV_OUTPUT_DIR = "csv_results"
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# Initialize LLM
# llm = OpenAI(temperature=0, api_key="your_openai_api_key")
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Function to execute a query and save results to a CSV file
def execute_and_save_to_csv(query: str, filename: str) -> str:
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
            
            # Save results to a CSV file
            csv_path = os.path.join(CSV_OUTPUT_DIR, filename)
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(columns)  # Write headers
                writer.writerows(rows)   # Write rows
            
            return csv_path
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to analyze a CSV file using LLM
def analyze_csv_with_llm(csv_path: str):
    try:
        # Read CSV content
        with open(csv_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        
        # Prepare a prompt for the LLM
        header = ", ".join(rows[0])
        data_sample = "\n".join([", ".join(row) for row in rows[1:6]])  # Use a sample of first 5 rows
        prompt = (
            f"Here is a dataset with the following columns: {header}.\n"
            f"Sample data:\n{data_sample}\n\n"
            "Analyze this dataset and provide insights, trends, and actionable recommendations."
        )
        
        # Generate insights
        insights = llm(prompt)
        return insights
    except Exception as e:
        return f"An error occurred during analysis: {str(e)}"

# General workflow for trends and patterns with LLM insights
def workflow_with_automated_insights(user_request: str, csv_filename: str):
    # Step 1: Generate SQL query
    sql_prompt_template = PromptTemplate(
        input_variables=["user_prompt"],
        template="Translate the following request into an SQL query:\n{user_prompt}"
    )
    sql_query = llm(sql_prompt_template.format(user_prompt=user_request))
    print(f"Generated SQL Query:\n{sql_query.content}")
    
    # Step 2: Execute query and save results to CSV
    csv_path = execute_and_save_to_csv(sql_query, csv_filename)
    # if "An error occurred" in csv_path:
    #     print(csv_path)
    #     return
    
    # print(f"Query results saved to {csv_path}")
    
    # Step 3: Analyze CSV with LLM
    # insights = analyze_csv_with_llm(csv_path)
    # print("\nGenerated Insights:")
    # print(insights)

# Example: Run the workflow
if __name__ == "__main__":
    user_requests = [
        ("Show me the top 5 best-selling products and their sales trends.", "top_products_trends.csv")
        # ("Identify the buying patterns for customers with high lifetime value.", "buying_patterns_high_ltv.csv"),
    ]
    
    for request, filename in user_requests:
        print(f"\nProcessing: {request}")
        workflow_with_automated_insights(request, filename)
