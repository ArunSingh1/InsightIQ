from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# Step 1: Database setup
DB_NAME = "olist"
DB_USERNAME = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"

# db_url = f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# db_url= 'postgresql+psycopg2://postgres:admin@localhost:5432/olist'
db_url = 'postgresql://postgres:admin@localhost:5432/olist'

engine = create_engine(db_url)

import psycopg2

def execute_query_tool(sql_query):
    try:
        # Connect to your postgres DB
        conn = psycopg2.connect(db_url)
        # Open a cursor to perform database operations
        cur = conn.cursor()

        # Execute the SQL query
        cur.execute(sql_query)

        # Fetch the result
        result = cur.fetchall()

        print(result)

        # rows = result.fetchall()
        # columns = result.keys()
            
        #     # Save results to a text file
        # with open("query_results.txt", "w") as file:
        #     file.write("\t".join(columns) + "\n")  # Write headers
        #     for row in rows:
        #         file.write("\t".join(map(str, row)) + "\n")  # Write rows

        # Close communication with the database
        cur.close()
        conn.close()

        return "Query executed successfully. Results saved to 'query_results.txt'."
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Step 2: Tool to execute SQL query
# def execute_query_tool(query: str) -> str:
#     try:
#         with engine.connect() as connection:
#             result = connection.execute(query)
#             rows = result.fetchall()
#             columns = result.keys()
            
#             # Save results to a text file
#             with open("query_results.txt", "w") as file:
#                 file.write("\t".join(columns) + "\n")  # Write headers
#                 for row in rows:
#                     file.write("\t".join(map(str, row)) + "\n")  # Write rows
            
#             return "Query executed successfully. Results saved to 'query_results.txt'."
#     except Exception as e:
#         return f"An error occurred: {str(e)}"

query_executor_tool = Tool(
    name="ExecuteQuery",
    func=execute_query_tool,
    description="Executes the provided SQL query on the database and saves the results."
)

# Step 3: LLM setup/
# llm = OpenAI(temperature=0, api_key="your_openai_api_key")
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Step 4: Custom function to confirm with the user
def confirm_and_execute(llm_query: str) -> str:
    print(f"Generated SQL Query:\n{llm_query}")
    approval = input("Do you want to execute this query? (yes/no): ").strip().lower()
    if approval == "yes":
        return query_executor_tool.run(llm_query)
    else:
        return "Query execution canceled."

# Step 5: Workflow with user confirmation
def workflow():
    # Get user input
    user_input = input("What do you want to know about the database? ").strip()
    
    # Generate SQL query
    sql_query_prompt = PromptTemplate(
        input_variables=["query"],
        template="Translate the following natural language query into an SQL query to be executed against a postgres db, include only queries related to postgres:\nQuery: {query}"
    )
    generated_query = llm(sql_query_prompt.format(query=user_input))

    sql_query = generated_query.content
    print(sql_query)
    print(type(sql_query))


    
    # Confirm and execute
    result = confirm_and_execute(sql_query)
    print(result)


# Step 6: Run the workflow
if __name__ == "__main__":
    workflow()
