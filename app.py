import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import json
from autogen import config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import UserProxyAgent
import autogen
from apify_client import ApifyClient


load_dotenv()
apify_api_key = os.getenv("APIFY_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
airtable_api_key = os.getenv("AIRTABLE_API_KEY")
config_list = config_list_from_json("OAI_CONFIG_LIST")

# ------------------ Create functions ------------------ #


# Function for google search
def google_search(search_keyword):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": search_keyword
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(search_keyword)
    print()
    print("RESPONSE:", response.text)
    return response.text


# Function for scraping
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])

    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=False
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


def web_scraping(objective: str, url: str):
    print("Scraping website...")
    print(f"{url} - {objective}")

    # Set your Apify API token here
    os.environ['APIFY_TOKEN'] = apify_api_key

    # Initialize the Apify client
    client = ApifyClient()

    # Define the JavaScript function to scrape the HTML content of the page
    page_function = """
    async function pageFunction(context) {
        const { request, log, jQuery } = context;
        return {
            url: request.url,
            html: document.documentElement.outerHTML,
        };
    }
    """

    # Define the data for the task with startUrls and pageFunction
    data = {
        "startUrls": [{"url": url}],
        "pageFunction": page_function
    }

    # Start the web scraping actor and wait for it to finish
    run = client.actor('apify/web-scraper').call(run_input=data)

    # Fetch and check the run details
    if run['status'] == 'SUCCEEDED':
        # Get the dataset items (the scraped data)
        dataset_items = client.dataset(run['defaultDatasetId']).list_items()

        if dataset_items:
            # Assuming the dataset items contain the HTML content
            html_content = dataset_items[0]['html']
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text()

            print("Scraped Content:", text)
            if len(text) > 10000:
                output = summary(objective, text)  # assuming 'summary' is a defined function
                return output
            else:
                return text
        else:
            return "No data scraped."
    else:
        print(f"Scraping failed with status {run['status']}")

    return "Scraping completed."


# Function for get airtable records
def get_airtable_records(base_id, table_id):
    print(f"get_airtables base_id:{base_id}, table_id:{table_id}")

    url = f"https://api.airtable.com/v0/{base_id}/{table_id}"

    headers = {
        'Authorization': f'Bearer {airtable_api_key}',
    }

    response = requests.request("GET", url, headers=headers)
    data = response.json()
    print(f"get airtable date after response {data}")
    return data


# Function for update airtable records
def update_single_airtable_record(base_id, table_id, id, fields):
    url = f"https://api.airtable.com/v0/{base_id}/{table_id}"

    headers = {
        'Authorization': f'Bearer {airtable_api_key}',
        "Content-Type": "application/json"
    }

    data = {
        "records": [{
            "id": id,
            "fields": fields
        }]
    }

    print(f"update before response {data}")

    response = requests.patch(url, headers=headers, data=json.dumps(data))
    data = response.json()
    print(f"update after response {data}")
    return data


# ------------------ Create agent ------------------ #


# Create user proxy agent
user_proxy = UserProxyAgent(name="user_proxy",
                            is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
                            human_input_mode="ALWAYS",
                            max_consecutive_auto_reply=1)

# Create researcher agent
researcher = GPTAssistantAgent(
    name="researcher",
    llm_config={
        "config_list": config_list,
        "assistant_id": "asst_aUU1DmziHsDOXoYZ1HFPH4Dx"
    }
)

researcher.register_function(
    function_map={
        "web_scraping": web_scraping,
        "google_search": google_search
    }
)

# Create research manager agent
research_manager = GPTAssistantAgent(
    name="research_manager",
    llm_config={
        "config_list": config_list,
        "assistant_id": "asst_LISgTzGrXoDdRC7W6DUV4YjQ"
    }
)


# Create director agent
director = GPTAssistantAgent(
    name="director",
    llm_config={
        "config_list": config_list,
        "assistant_id": "asst_L1ts5jT9pLDO9AxBvmAA7ajy",
    }
)

director.register_function(
    function_map={
        "get_airtable_records": get_airtable_records,
        "update_single_airtable_record": update_single_airtable_record
    }
)


# # Create group chat
groupchat = autogen.GroupChat(agents=[user_proxy, researcher, research_manager, director], messages=[], max_round=15)
group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})


# ------------------ start conversation ------------------ #
message = """
Research the funding stage/amount & pricing for each company in the list:
https://airtable.com/applkqpzwhqnsDv2K/tbl1VyGDInpVRNswt/viwnEf4Co9mxboGOJ?blocks=hide

first let the director see this
"""
user_proxy.initiate_chat(group_chat_manager, message=message)
