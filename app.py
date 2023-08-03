### Imports
import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, Tool, tool
from langchain import LLMMathChain
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
import os
import io
from contextlib import redirect_stdout
import sqlite3
from sqlite3 import Error
from typing import Optional, Type
from datetime import datetime
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path="langchain.db")

### - Layout components --
## I put these at the top because Streamlit runs from the top down and 
## I need a few variables that get defined here. 

## Layout configurations
st.set_page_config(
    page_title='Siva\'s App', 
    layout="wide",
    initial_sidebar_state='collapsed',
)
## CSS is pushed through a markdown configuration.
## As you can probably guess, Streamlit layout is not flexible.
## It's good for internal apps, not so good for customer facing apps.
padding_top = 10
st.markdown(f"""
    <style>
        .block-container, .main {{
            padding-top: {padding_top}px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

## UI Elements starting with the Top Graphics
col1, col2 = st.columns( [1,5] )
col1.image('AderasBlue2.png', width=70)
col1.image('AderasText.png', width=70)
col2.title('Simple LangChain SQL Example')
st.markdown('---')
## Add a sidebar
with st.sidebar: 
    show_detail = st.checkbox('Show Details')
    llm_model = st.selectbox('Select Model', ['gpt-4', 'gpt-3.5-turbo'])
    st.markdown("---")
    tz = st.container()


### --- A little housekeeping ---
## Creating the SQLite connection information. 
## BTW, Streamlit is stateless and all this runs each time the page is drawn.
conn = None
try:
    conn = sqlite3.connect('content/chinook.sqlite')
    cur = conn.cursor()
except Error as e:
    print(e)

## I need a function that will extract the SQL statement from the response from ChatGPT.
## Sometimes it prefices it with strings that deliminates the SQL and sometimes it adds 
## an explaination of th code...I just want the code. Prompt engineering works a little but not always.
def extract_select(text):
    start_index = text.upper().find("SELECT")
    if start_index == -1:
        return None
    end_index = text.find(";", start_index)
    if end_index == -1:
        return None
    sql_statement = text[start_index:end_index + 1]
    return sql_statement.strip()


# OpenAI Credentials
# I can't check the secrets file into GIT or OpenAI will terminate my key
if not os.environ["OPENAI_API_KEY"]:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = os.environ["OPENAI_API_KEY"]


### -- Next up is defining the LangChain chain ---

## This is a custom tool. LangChain has predefined tools, but I've had a few problems with it. 
## A blogger pointed out that LangChain tools are based upon prompts written for every conceivable
## use case. But your own tool will have prompts that are good at your specific use case...so I write my own.
## Extended classes isn't really normal Python but that's what you are doing here.
## You will see I put the table structure in the description. The description and your question, and 
## your prompt all become the prompt submitted to ChatGPT...basically, it's all Prompt Engineering.
class MySQLTool(BaseTool):
    name = "MySQLTool"
    description = """
This tool queries a SQLite database. It is useful for when you need to answer questions 
by running SQLite queries. Always indicate if your response is a "thought" or a "final answer". 
The following TABLES information is provided to help you write your sql statement.
Be sure to end all SQL statements with a semicolon. 
        
TABLES:
Album (AlbumId, Title, ArtistId)
Artist (ArtistId, Name)
Customer (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
Employee (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
Genre (GenreId, Name)
Invoice (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
MediaType (MediaTypeId, Name)
Playlist (PlaylistId, Name)
PlaylistTrack (PlaylistId, TrackId)
Track (TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
"""

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        # strip it down to the SELECT statement
        newquery = extract_select(query)

        ### Uncomment if you want the query printed on the page
        # st.markdown("#### Query Being Executed:")
        # st.markdown(newquery)

        # I previously defined the cursor
        try:
            cur.execute(newquery)
            results = cur.fetchall()
        except Error as e:
            results = "Error running query"
        
        return results  
    
    # I'm not using async but you could if you want the prompts to update 
    # the user screen as it progresses.
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


## Defining which model to use. We can discuss why you might want to use older models.
## There is a model selected for the overall chain and each tool has it's own.
## In theory you could use different models for each, but GPT-4 is the best...but slowest
llm = ChatOpenAI(model=llm_model, temperature=0, verbose=False)

## Besides SQL, I defined an agent that can perform mathematics.
## I'm just tossing this in to demonstrate chains are more than one tool.
llm_math = LLMMathChain.from_llm(llm=llm, verbose=False)
# palchain = PALChain.from_math_prompt(llm=llm, verbose=True)

## Now we define the tools that will be used in the chain. They look different
## because our custom tool already has thse properties defined in the package.
tools = [
    Tool(
        name="Calculator",
        func=llm_math.run,
        description="This tool is good at solving complex word math problems. Input should be a fully worded hard word math problem."

# Use the following format:"
    ),
    MySQLTool()]

## Now we define the agent
search_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    return_intermediate_steps=False,
)

## One more package...this runs when you enter a question and hit [ENTER]
## It runs everything and then finishes paining the rest of the page
def run_prompt(myquestion):
    # I'm using a chat approach so the questions and answers scroll down the page as you use the app.
    # It handles all the writing that occurs after you hit enter.
    st.session_state.messages.append({"role": "user", "content": myquestion})
    st.chat_message("user").write(myquestion)

    # I probably didn't need to put the database info in another prompt.
    template=f"""You are a DBA helping the user write and run ANSI SQL queries.
    Using the DATABASE INFORMATION below, provide a SQL query to the "QUESTION" below. 
    Be sure to end all SQL statements with a semicolon.

    DATABASE INFORMATION:
    Album (AlbumId, Title, ArtistId)
    Artist (ArtistId, Name)
    Customer (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
    Employee (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
    Genre (GenreId, Name)
    Invoice (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
    InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
    MediaType (MediaTypeId, Name)
    Playlist (PlaylistId, Name)
    PlaylistTrack (PlaylistId, TrackId)
    Track (TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)

    QUESTION:
    {myquestion}
    """

    # If detail is requested it captures the stdout.
    if show_detail:
        f = io.StringIO()
        with redirect_stdout(f):
            with st.spinner("Processing..."):
                response = search_agent.run(template)
    else:
        with st.spinner("Processing..."):
            response = search_agent.run(template)
    
    st.session_state.messages.append({"role": "assistant", "content": response})    
    st.chat_message('assistant').write(response)

    # If detail is requested the stdout is printed it in a collapsable region.
    if show_detail:
        with st.expander('Details', expanded=False):
            s = f.getvalue()
            st.write(s)

### -- Let's get back to building the web page --
## First run populates the session state with a benign message
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

## History Q&As are printed
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

## Now we ask for a question
if prompt := st.chat_input(placeholder="Ask a query-like question?"):
    start = datetime.now()
    tz.write("Start: "+str(start)[10:])
    run_prompt(prompt)
    tz.write("End: "+str(datetime.now())[10:])
    tz.write("Duration: "+str(datetime.now() - start))
