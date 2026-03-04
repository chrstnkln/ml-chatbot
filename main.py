print("Importing...")

"""
TODO:
Bisschen rumspielen mit Agentenarchitektur. SWARM spannend. Supervisor
DSR Prozessschritte schreiben
KISS!
"""

from dataclasses import dataclass
from typing import (
    Literal,
    Optional,
    Annotated
)

import httpx
import os
import requests
import urllib3
import uuid
import logging
import datarobot as dr
import colorama
import json
import random 
import string
import pandas as pd
import smolagents as smol

from dotenv import load_dotenv
from langchain_core.tools import tool as lc_tool, InjectedToolCallId
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, InjectedState
from pydantic import BaseModel, Field
from smolagents import tool, CodeAgent
from snowflake.core import Root
from snowflake.core.cortex.lite_agent_service._generated.models import AgentRunRequest
from snowflake.snowpark import Session

from mempd import InMemoryPandas


logging.getLogger('datarobot').setLevel(level=logging.WARNING)
logging.getLogger('urllib3').setLevel(level=logging.WARNING)
logging.getLogger('requests').setLevel(level=logging.WARNING)

urllib3.disable_warnings()

PYPPETEER_CHROMIUM_REVISION = '1263111'
os.environ['PYPPETEER_CHROMIUM_REVISION'] = PYPPETEER_CHROMIUM_REVISION

print("Loading environment...")

load_dotenv(override=True)

print("Connecting to DataRobot...")

DATAROBOT_ENDPOINT = '###'
DATAROBOT_API_KEY = '###'

dr.client.Client(
    endpoint=DATAROBOT_ENDPOINT,
    token=DATAROBOT_API_KEY,
    ssl_verify=False,
    connect_timeout=20,
    max_retries=10
)

print("Connecting to Snowflake...")

sf_connection_params = {
    "private_key_file": "###",
    "user":"###",
    "account":"###",
    "database":"###",
    "role":"###",
    "schema":"###",
}

sf_session = Session.builder.configs(sf_connection_params).create()
sf_root = Root(connection=sf_session)

base_url = "###"
iam_client_id = os.getenv("IAM_CLIENT_ID")
iam_access_token = os.getenv("IAM_ACCESS_TOKEN")
cert_path = "###"

auth_url = "###"
auth_headers = {
    "Content-Type": "application/x-www-form-urlencoded"
}
auth_data = {
    "grant_type": "client_credentials",
    "client_id": iam_client_id,
    "client_secret": iam_access_token,
    "scope": "machine2machine"
}

print("Acquiring LLM access token...")

auth_response = requests.post(auth_url, headers=auth_headers, data=auth_data)

if auth_response.status_code != 200:
    print(f"Error: {auth_response.status_code} - {auth_response.text}")

access_token = auth_response.json()["access_token"]

llm = ChatOpenAI(
    #temperature=0.1,
    model="openai/gpt-4.1",
    temperature=0.0,
    # model="anthropic/claude-3-7-sonnet",
    openai_api_key=access_token,
    http_client=httpx.Client(
        verify=cert_path,
        headers={
            "Accept": "application/json",
            "x-apikey": iam_client_id,
            "Content-Type": "application/json",
        },
    ),
    base_url=base_url,
    max_completion_tokens=4000
)

print("Initializing smolagents model...")

smol_model = smol.OpenAIModel(
    model_id="gpt-4.1",
    api_base=base_url,
    api_key=access_token,
    client_kwargs={
        "http_client": httpx.Client(
            verify=cert_path,
            headers={
                "Accept": "application/json",
                "x-apikey": iam_client_id,
                "Content-Type": "application/json",
        })},
    temperature=0.0,
)


mempd = InMemoryPandas()


colorama.init()


def log(agent, type, value, params):
    print(f"{colorama.Style.DIM}[LOG]    {agent}{(22 - (len(agent))) * ' '}{type}{(12 - len(type)) * ' '}{value}{(22 - len(value)) * ' '}{str(params)}{colorama.Style.RESET_ALL}")


def sf_create_agent_run_request(request: str) -> AgentRunRequest:
    return AgentRunRequest(
        messages=
        [
            {
                "role": "user",
                "content":
                [
                    {
                        "type": "text",
                        "text": f"""You are an advanced data retrieval agent, that is supposed to find and transform relevant data for a request for later data retrieval. Perform JOINS, aggregation, transformation or filtering of tables IF NECESSARY, to retrieve the requested data.
The query HAS TO BE Snowflake compatible and executable with Snowpark!
RETURN ONLY COLUMNS AND DATA THAT ARE REQUIRED AND REQUESTED. E.g. if only one feature is requested, return only one in the query! Always include the schema in the query!
Check beforehand if the query returns more than 1,000 entries and LIMIT ONLY IF NECESSARY. Inform inside the YAML semantics in case the data is limited to 1,000 rows!
NOTE: You CANNOT combine columns of type TIME and DATE with "+" or TIMESTAMP function in the final query, use e.g. DATE_PART! Make sure to synchronize timezones when working with timestamps of different zones!
A query MAY NOT return more than 1,000 rows. Your response MAY ONLY consist of one final Snowflake query to retrieve the requested data with snowpark later on, alongside a generated YAML semantics file where e.g. resulting table and columns of the query are explained. Only return on data that is actually required for the use case.\n\n## Request:\n{request}?"""
                    }
                ]
            }
        ]
    )


@lc_tool
def t_requires_context(
    requirement_description: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """
    A tool that can be called whenever further context, information or input
    is required, e.g. by a human, to fulfill a request

    Arg:
        requirement_description:    A detailed description of the context, information or input that is required

    Return:
        str:   The description returned to the main AutoML Agent to try and retrieve the information
    """
    log('GLOBAL', 'TOOL_CALL', 'requires-context', {'requirement_description': requirement_description})

    tool_message = ToolMessage(
        content=state.values['messages'][-1].content,
        name='_t_analyze_dataset',
        tool_call_id=tool_call_id
    )

    return requirement_description

    
@tool
def smol_load_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Loads a dataset by ID as a tuple, containing a pandas dataframe


    Args:
        dataset_id: ID of the dataset

    Returns:
        [pandas.DataFrame:   Loaded pandas dataframe
    """
    return mempd.get(dataset_id.lower())

@tool
def smol_save_dataset(df: pd.DataFrame) -> str:
    """
    Saves and persists a new dataset using a pandas dataframe.

    Args:
        df:   Dataframe to be saved

    Returns:
        str:   ID of the newly created dataset
    """
    dsid = str(uuid.uuid4()).lower().split("-")[0]
    mempd.add(id=dsid, df=df)

    writer = get_stream_writer()
    writer({"new_dataset": dsid})

    return dsid


class DataAnalysisAgent:
    SYSTEM_PROMPT = """
You are a data agent that has the ability to retrieve, analyze and transform datasets,
including feature analysis. When retrieving data, don't tell what data you currently need, but what the final dataset should look like.
Your goal is to fulfill the analysis request and provide a detailed response for another AI agent,
that will turn your response into an answer.
If you used a dataset or multiple datasets, include the ID(s) of the used datasets in your response.
The correct capitalization of column names of datasets is IMPORTANT!
DO NOT ASSUME.
"""

    def __init__(self):
        self._graph, self._graph_config = self._build_graph()

    def _build_graph(self):
        graph_builder = StateGraph(MessagesState)

        data_tn = ToolNode([
            DataAnalysisAgent._t_retrieve_datasets_by_description,
            DataAnalysisAgent._t_analyze_dataset,
            DataAnalysisAgent._t_transform_dataset,
        ])

        graph_builder.add_node('process', self._n_process)
        graph_builder.add_node('retrieve-dataset', data_tn)
        graph_builder.add_node('analyze-dataset', data_tn)
        graph_builder.add_node('transform-dataset', data_tn)

        graph_builder.add_edge(START, 'process')

        graph_builder.add_conditional_edges(
            'process',
            self._c_should_continue,
            {
                'retrieve-dataset': 'retrieve-dataset',
                'analyze-dataset': 'analyze-dataset',
                'transform-dataset': 'transform-dataset',
                END: END
            }
        )

        graph_builder.add_edge('retrieve-dataset', 'process')
        graph_builder.add_edge('analyze-dataset', 'process')
        graph_builder.add_edge('transform-dataset', 'process')

        checkpointer = InMemorySaver()
        graph = graph_builder.compile(checkpointer=checkpointer)
        config = { 'configurable': { 'thread_id': ''.join(random.choice(string.ascii_letters) for _ in range(10))  } }

        return graph, config
    
    def _draw_graph(self):
        print("Drawing DataAnalysisAgent graph...")
        print(self._graph.get_graph().draw_ascii())

    def _n_process(self, state: MessagesState):
        writer = get_stream_writer()
        writer({"node_log": "[DA] (node) process"})
        writer({"alog": f"[DA] (process) Processing for data agent"})

        messages = state["messages"]
        response = llm \
            .bind_tools([
                DataAnalysisAgent._t_retrieve_datasets_by_description,
                DataAnalysisAgent._t_analyze_dataset,
                DataAnalysisAgent._t_transform_dataset,
            ]) \
            .invoke(messages)
        
        print(response)

        return {"messages": [response]}
    
    @lc_tool
    def _t_retrieve_datasets_by_description(data_description: str) -> tuple[str, str]:
        """
        Hands off to a data retrieval agent that can load and provided a ready transformed dataset, so that the user can work with it.
        Usually, this tool needs to be called only once for initial retrieval.
        This tool CAN PERFORM data JOINS, transformation and aggregation on retrieval. NO NEED to load multiple datasets for later transformation.
        ALWAYS call this tool when some sort of dataset is requested to be worked with or loaded or whatever.
        If you don't know the exact use case, why and which data is needed: DO NOT ASSUME! Only work with the information that you were provided!

        Arg:
            data_description:   A detailed description of the data to be retrieved alongside a purpose for retrieval, if provided (str)

        Returns:
            tuple[str (Dataset ID), str (Semantic YAML)]:    IDs and semantic description of the dataset
        """
        log('AutoMLAgent', 'TOOL_CALL', 'retrieve-dataset', {'data_description': data_description})

        writer = get_stream_writer()
        writer({"node_log": "[DA] (tool) retrieve-dataset"})
        writer({"alog": f"[DA] (retrieve-ds-by-desc) Retrieving datasets... Data description: {data_description}"})

        sf_resp = None

        print(f"Calling Snowflake Cortex Agent with request:\n{data_description}")

        for event in sf_root.cortex_agent_service.run(sf_create_agent_run_request(data_description)).events():
            if event.event == "response":
                for k, v in json.loads(event.data).items():
                    if k != "content":
                        continue
                    for itm in v:
                        for k2, v2 in itm.items():
                            if k2 == "text":
                                sf_resp = v2

        if sf_resp is None:
            writer({"alog": f"[DA] (retrieve-ds-by-desc) Unable to find relevant data..."})
            return "Unable to find relevant data"
        
        print (sf_resp)
        
        class DatasetQuery(BaseModel):
            query: str = Field(description="SQL query")
            semantic_yaml: str = Field(description="Semantic table and column description in YAML format")

        class DatasetQueryList(BaseModel):
            queries: list[DatasetQuery]

        writer({"alog": f"[DA] Raw Snowflake agent response:\n{sf_resp}"})
        print("RE-STRUCTURING...")

        response = llm \
            .with_structured_output(schema=DatasetQuery) \
            .invoke(f"Structure the following LLM output to the requested output format, including an executable SQL query string and a semantic description of the result as YAML:\n\n{sf_resp}")
        
        print(response)

        dsid = str(uuid.uuid4()).lower().split("-")[0]
        print(dsid)

        try:
            dataf = DataAnalysisAgent._t_retrieve_raw_data_by_sql(response.query)
        except Exception as e:
            print(e)

            return f"An error occurred trying to retrieve the dataset by SQL: {e}, Query: {response.query}"
        
        mempd.add(
            id=dsid,
            df=dataf,
            semantics=response.semantic_yaml
        )
        writer({"alog": f"[DA] (retrieve-ds-by-desc) Successfully added dataset ID: {dsid}"})
        print(dsid)
        return (dsid, mempd.get_with_semantics(dsid)[1])
        
    def _t_retrieve_raw_data_by_sql(sql_query: str, retry=0, errlog=""):
        """
        A data retrieval tool that retrieves datasets using a VALID SQL query as input, so that the user can work with it.

        Arg:
            sql_query:   Snowflake SQL query to retrieve the data

        Returns:
            pandas.DataFrame || None:    Retrieved data as a pandas DataFrame, if successful
        """
        print(f"Executing SQL query using Snowpark:\n{sql_query}")
        writer = get_stream_writer()
        writer({"alog": f"[DA] (retrieve-ds-by-sql) Executing SQL query: {sql_query}"})
        sql_query = sql_query.strip()
        if sql_query[-1] == ';': sql_query = sql_query[:-1]
        print(sql_query)
        # df = None
        # for b in sf_session.sql(query=sql_query).to_pandas_batches():
        #     if df is None: df = b
        #     else: df = df + b
        RETRY_LIMIT = 3

        try:
            df = sf_session.sql(str(sql_query)).to_pandas()
        except Exception as e:
            print(f"SQL ERROR: {e}")
            errlog += f"Retry #{retry}: The following error occurred while executing Snowflake SQL Queries: {e}\n\nQuery:{sql_query}\n\n"
            response = llm.invoke(errlog +
                                f"Fix the query, so that it works. ONLY RESPOND with the new Snowflake SQL Query in raw executable plain text!")
            print(response)
            if retry == RETRY_LIMIT:
                raise e
            return DataAnalysisAgent._t_retrieve_raw_data_by_sql(response.content, retry + 1, errlog)
        print(df)
  
        # sf_session.rollback()
        return df
        
            
    @lc_tool
    def _t_transform_dataset(
        dataset_ids: list[str],
        transformation_instruction: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ):
        """
        A tool that analyzes and transforms already existing datasets, make changes and save it as a new dataset.
        This tool can ALSO perform an ANALYSIS of the datasets. So NO PRIOR ANALYSIS TOOL REQUIRED.
        Make sure you include all possibly helpful datasets for transformation.
        Datasets do not necessarily have to be connected or dependent in any way in order to be useful for transformation.
        
        This tool can (besides other things) be used to prepare and transform a dataset before modeling.

        Arg:
            transformation_instruction: The inteded goal of the transformation
            dataset_ids: List of IDs of the datasets to be used for transformation
        Return:
            str:    Description of the transformation including the newly created dataset ID
        """

        log('DataAnalysisAgent', 'TOOL_CALL', 'transform-dataset', {'dataset_ids': dataset_ids, 'transformation_instruction': transformation_instruction})

        writer = get_stream_writer()
        writer({"node_log": "[DA] (tool) transform-dataset"})

        code_agent = CodeAgent(
            additional_authorized_imports=["pandas"],
            tools=[smol_load_dataset, smol_save_dataset],
            model=smol_model,
            max_print_outputs_length=60
        )

        return code_agent.run(
            f"Transform the datasets to a new dataset, if successful save it and return a new dataset ID alongside a description of how and what you did. "
            f""
            f"The YAML string contains semantic definitions for the datasets. The columns and meanings defined in the YAML are all you need! Don't look elsewhere."
            f"If provided, you can use multiple datasets and join them, if necessary."
            f"Be aware of timestamps with different timezones."
            f"Instruction: {transformation_instruction}",
            additional_args={'datasets': [{'id': i, 'semantic_yaml': mempd.get_with_semantics(i)[1]} for i in dataset_ids]},
            max_steps=40
        )
    
    @lc_tool
    def _t_analyze_dataset(
        dataset_id: str,
        analysis_instruction: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ):
        """
        A tool that can be used to get information about a dataset.
        DO NOT USE IF DATASET TRANSFORMATION IS THE GOAL.

        Arg:
            analysis_instruction: A detailed description of the analysis to be performed
            dataset_id: ID of the dataset

        Return:
            Any:    Resulting object containing the results of the analysis
        """

        log('DataAnalysisAgent', 'TOOL_CALL', 'analyze-dataset', {'dataset_id': dataset_id, 'analysis_instruction': analysis_instruction})

        writer = get_stream_writer()
        writer({"node_log": "[DA] (tool) analyze-dataset"})

        code_agent = CodeAgent(
            #stream_outputs=True,
            additional_authorized_imports=["pandas"],
            tools=[smol_load_dataset],
            model=smol_model,
            max_print_outputs_length=20
        )

        result = code_agent.run(
            f"Perform data analysis to fulfill the following instructions and return results as a dict or list. "
            f"You are provided a dataset alongside a semantic description of it in form of a YAML string. (Be aware that the datasets may contain many rows): {analysis_instruction}",
            additional_args={'dataset_id': dataset_id, 'semantic_yaml': mempd.get_with_semantics(dataset_id)[1]}
        )

        return result
    
    
    def _c_should_continue(self, state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "analyze-dataset"
        return END
    
    def invoke(self, data_instruction: str):
        log('DataAnalysisAgent', 'INVOKED', '', {'data-instruction': data_instruction})

        for mode, value in self._graph.stream(
            {
                'messages':
                [
                    SystemMessage(DataAnalysisAgent.SYSTEM_PROMPT),
                    HumanMessage(content=data_instruction)
                ],
            },
            config=self._graph_config,
            stream_mode=["custom"]
        ):
            yield mode, value

        #return self._graph.get_state(self._graph_config)
    

data_agent = DataAnalysisAgent()


class ModelingAgent:
    SYSTEM_PROMPT = """
You are a AutoML modeling agent that has the ability to create, analyze and interact with machine learning projects and their ML-models.
Your goal is to fulfill all requests related to modeling (e.g. model creation, information about projects / models, and more!)
DO NOT ASSUME.
If you are can't fulfill the request, BE CLEAR about it.
My career depends on your expertise, so BE GOOD!

==========

Dataset ID (if available): {dataset_id}
"""
    analysis_agent = DataAnalysisAgent()

    def __init__(self):
        self._graph, self._graph_config = self._build_graph()

    def _build_graph(self):
        graph_builder = StateGraph(MessagesState)

        # analyze_tn = ToolNode([
        #     DataAnalysisAgent._t_analyze_features
        # ])

        tn_model = ToolNode([
            ModelingAgent._t_create_models,
            ModelingAgent._t_get_models,
            ModelingAgent._t_list_projects,
            ModelingAgent._t_get_project,
            ModelingAgent._t_deploy_model,
            ModelingAgent._t_activate_deployment,
            t_requires_context
        ])

        graph_builder.add_node('process', self._n_process)
        graph_builder.add_node('create-models', tn_model)
        graph_builder.add_node('get-models', tn_model)
        graph_builder.add_node('list-projects', tn_model)
        graph_builder.add_node('get-project', tn_model)
        graph_builder.add_node('deploy-model', tn_model)
        graph_builder.add_node('activate-deployment', tn_model)
        graph_builder.add_node('requires-context', tn_model)

        # graph_builder.add_node('feature-selection', analyze_tn)

        graph_builder.add_edge(START, 'process')

        graph_builder.add_conditional_edges(
            'process',
            self._c_should_continue,
            {
                'list-projects': 'list-projects',
                'get-project': 'get-project',
                'create-models': 'create-models',
                'get-models': 'get-models',
                'deploy-model': 'deploy-model',
                'activate-deployment': 'activate-deployment',
                'requires-context': 'requires-context',
                END: END
            }
        )

        graph_builder.add_edge('create-models', 'process')
        graph_builder.add_edge('get-models', 'process')
        graph_builder.add_edge('list-projects', 'process')
        graph_builder.add_edge('get-project', 'process')
        graph_builder.add_edge('deploy-model', 'process')
        graph_builder.add_edge('activate-deployment', 'process')

        graph_builder.add_edge('requires-context', END)

        checkpointer = InMemorySaver()
        graph = graph_builder.compile(checkpointer=checkpointer)
        config = { 'configurable': { 'thread_id': ''.join(random.choice(string.ascii_letters) for _ in range(10)) } }

        return graph, config
    
    def _draw_graph(self):
        print("Drawing ModelingAgent graph...")
        print(self._graph.get_graph().draw_ascii())

    def _n_process(self, state: MessagesState):
        writer = get_stream_writer()
        riter = get_stream_writer()
        writer({"node_log": "[AA] (node) process"})
        writer({"alog": f"[AA] (process) Processing for modeling agent"})

        messages = state["messages"]
        response = llm \
            .bind_tools([
                ModelingAgent._t_create_models,
                ModelingAgent._t_get_models,
                ModelingAgent._t_list_projects,
                ModelingAgent._t_get_project,
                ModelingAgent._t_deploy_model,
                ModelingAgent._t_activate_deployment,
                t_requires_context
            ]) \
            .invoke(messages)

        return {"messages": [response]}

    @lc_tool
    def _t_create_models(
        dataset_id: str,
        project_name: str,
        use_case: str,
        is_unsupervised: bool,
        unsupervised_type: Optional[Literal['cluster', 'anomaly']],
        target: Optional[str]
    ):
        """
        A tool that automatically creates and trains ML models using a training dataset.
        In order to call this tool, a HUMAN NEEDS TO PROVIDE the exact PROJECT NAME! DO NOT choose a project name yourself! Use the require_context tool to ask for a project name, if not provided!
        This tool will only start the modeling process, but it will not wait until modeling is finished.

        Arg:
            dataset_id:         ID of the training dataset
            project_name:       Name of the machine learning project. This MUST ALWAYS be PROVIDED BY a HUMAN. If not specified, ask for it!
            is_unsupervised:    Specifies if the machine learning task is an unsupervised task.
            unsupervised_type:  Specifies the unsupervised learning type ('anomaly' or 'cluster'). Only necessary if it is an unsupervised task.
            target:             The target variable of the dataset to be predicted. Must be null if it is an unsupervised task. If it is not null, it MUST be a valid and existing feature / column of the underlying dataset

        Returns:
            str:    ID of newly created project, containing the machine learning models.
        """
        log(
            'ModelingAgent', 'TOOL_CALL', 'create-models',
            {
                'dataset_id': dataset_id,
                'project_name': project_name,
                #'use_case': use_case,
                'is_unsupervised': str(is_unsupervised),
                'unsupervised_type': str(unsupervised_type),
                'target': target
            }
        )

        writer = get_stream_writer()
        writer({"alog": f"[AA] (create-models) Model creation started... project_name: {project_name} unsupervised: {is_unsupervised} unsupervised_type: {unsupervised_type} target: {target}"})
        writer({"node_log": f"[AA] (tool) create-models"})

        # ds = dr.Dataset.get(dataset_id=dataset_id)

        # project = ds.create_project(project_name=project_name)

        # project.analyze_and_model(
        #     target=target,
        #     unsupervised_mode=is_unsupervised,
        #     unsupervised_type=unsupervised_type
        # )

        # return project.id

        ds = dr.Dataset.create_from_in_memory_data(
            data_frame=mempd.get(dataset_id),
            fname=str(dataset_id)
        )

        project = ds.create_project(project_name=project_name)
        project.analyze_and_model(
            target=target,
            unsupervised_mode=is_unsupervised,
            unsupervised_type=unsupervised_type
        )

        return project.id

        # return "691ef3a0ce59935123859bca"

    @lc_tool
    def _t_list_projects(search_name: Optional[str], search_id: Optional[str]):
        """
        A tool that lists already existing ML-projects

        Arg:
            search_name: Search string for the name, optional
            search_id: Search ID for project, optional
        Returns:
            dict<str (project ID), str (project name)> Dictionary of all projects
        """
        writer = get_stream_writer()
        writer({"alog": f"[AA] (get_project) Retrieving project(s)... search_name: {search_name} search_id: {search_id}"})
        writer({"node_log": f"[AA] (tool) list-projects"})
        return {p.id: p.project_name for p in dr.Project.list()}


    @lc_tool
    def _t_get_project(search_name: Optional[str], search_id: Optional[str]):
        """
        A tool that gets a peoject based on ID or name

        Arg:
            search_name: Search string for the name, optional
            search_id: Search ID for project, optional
        Returns:
            project_id | None: Project ID or none
        """
        writer = get_stream_writer()
        writer({"alog": f"[AA] (get_project) Retrieving project(s)... search_name: {search_name} search_id: {search_id}"})
        writer({"node_log": f"[AA] (tool) get-project"})

        if search_id:
            return dr.Project.get(search_id)
        
        params = {'project_name': search_name} if search_name else None
        projs = dr.Project.list(search_params=params)

        if len(projs) > 1:
            return f"Multiple projects found: {", ".join([p.project_name for p in projs])}"
        elif len(projs) == 0:
            return f"No projects found!"
        
        return projs[0].project_name
    
    @lc_tool
    def _t_get_models(project_id: str):
        """
        A tool that retrieves current information about all models that were or are being created for a specific project.
        This includes finished models and models that are currently being trained.
        The information contains model type, model family, model category, performance and quality metrics of the model and more.

        Arg:
            project_id: ID of the project

        Returns:
            list:   A list of models
        """
        writer = get_stream_writer()
        writer({"alog": f"[AA] (process) Retrieving existing models for project {project_id}..."})
        writer({"node_log": f"[AA] (tool) get-models"})

        project = dr.Project.get(project_id=project_id)

        finished_models: list[dr.models.Model] = project.get_models()
        unfinished_models: list[dr.models.ModelJob] = project.get_model_jobs()

        return [f"ID: {x.id} ({x.model_category}, {x.model_family}, {x.model_type})" for x in finished_models]
    
    def _t_deploy_model (
        project_id: str,
        model_id: str,
        version_name: str,
        registered_model_name: str,
        deployment_label: str,
        description: str
    ) -> str:
        """
        A tool that deploys a machine learning model of a certain project for productive use. USE ONLY if explicitly instructed to do so!
        The deployment will be created and it will be inactive.

        Arg:
            project_id: ID of the project
            model_id: ID of the model to be deployed
            version_name: Name of the specific version of this model (a.k.a. model package)
            registered_model_name: Name of the registered model (must be unique across the organization)
            deploment_label: Label of the deployment
            description: Decscription of the deployment

        Returns:
            str: ID of the ready deployment
        """
        writer = get_stream_writer()
        writer({"alog": f"[AA] (tool) Deploying model for {project_id} with id {model_id}..."})
        writer({"node_log": f"[AA] (tool) deploy-model"})

        model = dr.Model.get(
            project=project_id,
            model_id=model_id
        )

        registered_model_version = dr.RegisteredModelVersion.create_for_leaderboard_item(
            model_id=model.id,
            name=version_name,
            registered_model_name=registered_model_name
        )

        deployment = dr.Deployment.create_from_registered_model_version(
            model_package_id=registered_model_version.id,
            label=deployment_label,
            description=description
        )

        return deployment.id
    
    def _t_activate_deployment(self, deployment_id: str):
        """
        A tool that activates an existing deployment. ONLY USE IF EXICPLITLY INSTRUCTED. This will incur costs.

        Args:
            deployment_id: ID of the deployment
        """
        deployment = dr.Deployment.get(deployment_id)
        deployment.activate()

    def _c_should_continue(self, state: MessagesState):
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return 'get-models'
        return END

    # @lc_tool
    # def _t_determine_prediction_mode(data_description: str):
    #     """
    #     A data retrieval tool that semantically searches for relevant datasets in a predefined folder

    #     Arg:
    #         data_description:   A brief semantical description of the data to be retrieved (str)

    #     Returns:
    #         str:    ID of the dataset
    #     """
    #     log('AutoMLAgent', 'TOOL_CALL', 'retrieve-dataset', {'data_description': data_description})

    #     DATA_DIR = './datasets'

    #     ds = dr.Dataset.upload(os.path.join(DATA_DIR, 'patients.csv'))
    #     x =  dr.Project.create_from_dataset("")
    #     x = dr.Project()

    #     return ds.id

    def invoke(self, dataset_id: Optional[str], modeling_instruction: str, messages: Optional[list[BaseMessage]] = None):
        log('ModelingAgent', 'INVOKED', '', {'modeling_instruction': modeling_instruction})

        if messages is None:
            messages = []

        for mode, value in self._graph.stream(
            {
                'messages':
                [
                    SystemMessage(ModelingAgent.SYSTEM_PROMPT.format(dataset_id=dataset_id)),
                    HumanMessage(content=modeling_instruction)
                ],
            },
            config=self._graph_config,
            stream_mode=["custom"]
        ):
            yield mode, value

        # return self._graph.get_state(self._graph_config)


class AutoMLAgent:
    SYSTEM_PROMPT = """
You are a helpful data science chat assistant that has the ability search for data,
analyze data and generate machine learning models from the data.

You DO NOT assume your own capabilities.
You DO NOT assume values.
If you are not sure in your response, DO NOT assume.
You keep your answers short and concise.
Your are supposed to answer, ALWAYS!
Your response MUST be formatted nicely in Markdown!
If a dataset or multiple datasets were used, include their IDs as metadata at the bottom of your response.
The correct capitalization of column names of datasets is IMPORTANT!
My career depends on your expertise, so BE GOOD!
"""
    _modeling_agent = ModelingAgent()

    def __init__(self):
        self._graph, self._graph_config = self._build_graph()
        self._draw_graph()

    def _build_graph(self):
        graph_builder = StateGraph(MessagesState)

        process_tn = ToolNode([
            AutoMLAgent._t_data_agent_handoff,
            AutoMLAgent._t_modeling_agent_handoff,
        ])

        graph_builder.add_node('process', AutoMLAgent._n_process)
        graph_builder.add_node('data-agent-handoff', process_tn)
        graph_builder.add_node('automl-agent-handoff', process_tn)
        #graph_builder.add_node('analyze-features', analyze_tn)
        graph_builder.add_node('answer', AutoMLAgent._n_answer)

        graph_builder.add_node('data-agent', data_agent._graph)
        graph_builder.add_node('automl-agent', AutoMLAgent._modeling_agent._graph)

        graph_builder.add_edge(START, 'process')
        graph_builder.add_conditional_edges(
            'process',
            self._c_should_continue,
            {
                'data-agent-handoff': 'data-agent-handoff',
                'automl-agent-handoff': 'automl-agent-handoff',
                'answer': 'answer',
            }
        )

        graph_builder.add_edge('data-agent-handoff', 'data-agent')
        graph_builder.add_edge('data-agent', 'process')

        graph_builder.add_edge('automl-agent-handoff', 'automl-agent')
        graph_builder.add_edge('automl-agent', 'process')

        graph_builder.add_edge('answer', END)

        # graph_builder.add_conditional_edges(
        #     'analyze-dataset',
        #     AutoMLAgent._c_analysis_should_continue,
        #     {
        #         'analyze-features': 'analyze-features',
        #         'process': 'process'
        #     }
        # )

        #graph_builder.add_edge('analyze-features', 'analyze-dataset')

        checkpointer = InMemorySaver()
        graph = graph_builder.compile(checkpointer=checkpointer)
        config = { 'configurable': { 'thread_id': ''.join(random.choice(string.ascii_letters) for _ in range(10))  } }

        return graph, config
    
    def _draw_graph(self):
        print("Drawing AutoMLAgent graph...")
        #print(self._graph.get_graph(xray=1).draw_ascii())
        self._graph.get_graph(xray=2).draw_mermaid_png(output_file_path="graph2.png")

    def _n_process(state: MessagesState):
        messages = state["messages"]
        response = llm \
            .bind_tools([
                AutoMLAgent._t_data_agent_handoff,
                AutoMLAgent._t_modeling_agent_handoff,
            ]) \
            .invoke(messages)

        writer = get_stream_writer()
        writer({"node_log": "[SV] (node) process"})
        writer({"alog": "[SA] (process) Processing for supervision"})

        return {"messages": [response]}
    
    def _n_answer(state: MessagesState):
        print("Answering...")
        messages = state["messages"]
        response = ''

        writer = get_stream_writer()
        writer({"node_log": "[SV] (node) answer"})
        writer({"alog": "[SA] (answer) Generating answer..."})

        print(colorama.Style.BRIGHT)
        for chunk in llm.stream(messages):
            if chunk.content:
                print(chunk.content, end='', flush=True)
                writer({"answer_chunk": chunk.content})
                response += chunk.content

        print(colorama.Style.RESET_ALL)

        return {"messages": [response]}
    
    @lc_tool
    def _t_data_agent_handoff(
        data_instruction: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ):
        """
        A tool that temporarily hands off to a data agent.
        This agent can load, retrieve, analyze and prepare (transform) datasets.
        If a final dataset was already loaded and is ready, use the modeling agent to create and train a model, NOT THIS TOOL.
        ALWAYS hand off to this tool if the inquiry bs the human has something to do with data.

        Arg:
            data_instruction:   Description of the analysis task that is to be performed
        
        Return:
            Any:   Results by the data agent
        """
        log('AutoMLAgent', 'TOOL_CALL', 'data-agent-handoff', {'data_instruction': data_instruction})

        # messages = state["messages"]
        # response = llm \
        #     .bind_tools([
        #         AutoMLAgent._t_analyze_features,
        #     ]) \
        #     .invoke(messages[:-1])

        writer = get_stream_writer()
        writer({"node_log": "[SV] (tool)\tdata-agent-handoff"})
        writer({"alog": f"[SV] (data-agent-handoff) Hanfing off to data agent... Data instruction: {data_instruction}"})

        for mode, value in data_agent.invoke(     
            data_instruction=data_instruction
        ):
            if mode == "custom":
                writer(value)

        data_agent_state = data_agent._graph.get_state(data_agent._graph_config)

        print(f"\n\n{data_agent_state.values['messages'][-1].content}\n\n")

        tool_message = ToolMessage(
            content=data_agent_state.values['messages'][-1].content,
            name='_t_data_agent_handoff',
            tool_call_id=tool_call_id
        )

        #update = {'messages': state['messages'] + [tool_message, response]}
        return {'messages': state['messages'] + [tool_message]}
        
    @lc_tool
    def _t_modeling_agent_handoff(
        modeling_instruction: str,
        dataset_id: Optional[str],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ):
        """
        A tool that temporarily hands off to the modeling agent.
        The modeling agent is capable of creating, training, getting information and interacting with machine learning projects and ML models.
        ALWAYS make sure, that the dataset for modeling contains ONLY RELEVANT ROWS! Because training will use the entire dataset! Otherwise, transform the dataset first.

        Arg:
            modeling_instruction:  Elaborate description of the task for the modeling agent
            Optional[dataset_id]: ID of the training dataset, if it is needed and exists

        Returns:
            Optional[str]:    ID of the machine learning model OR null if further processing is required
        """
        log('SupervisorAgent', 'TOOL_CALL', 'automl-agent-handoff', {'dataset_id': dataset_id, 'modeling_instruction': modeling_instruction})

        writer = get_stream_writer()
        writer({"node_log": "[SV] (tool) automl-agent-handoff"})
        writer({"alog": f"[SV] (automl-agent-handoff) Hanfing off to modeling agent... Modeling instruction: {modeling_instruction}"})
    
        for mode, value in AutoMLAgent._modeling_agent.invoke(
            dataset_id=dataset_id,
            modeling_instruction=modeling_instruction
        ):
            if mode == "custom":
                writer(value)

        modeling_agent_state = AutoMLAgent._modeling_agent._graph.get_state(
            AutoMLAgent._modeling_agent._graph_config
        )

        tool_message = ToolMessage(
            content=modeling_agent_state.values['messages'][-1].content,
            name='_t_modeling_agent_handoff',
            tool_call_id=tool_call_id
        )

        return {'messages': state['messages'] + [tool_message]}
    
    def _c_should_continue(self, state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        
        if last_message.tool_calls:
            return "automl-agent-handoff"
        return 'answer'
    
    def invoke(self, prompt: str):
        log('AutoMLAgent', 'INVOKED', '', {'prompt': prompt})
        for graph, mode, value in self._graph.stream(
            { 'messages':
                [
                    SystemMessage(AutoMLAgent.SYSTEM_PROMPT),
                    HumanMessage(content=prompt)
                ]
            },
            stream_mode=['messages', 'tasks', "custom"],
            config=self._graph_config,
            subgraphs=True
        ):
            #print(mode, value)
            yield graph, mode, value


# def run_agent():
#     agent = AutoMLAgent()

#     while True:
#         try:
#             prompt = input(f'\n{colorama.Fore.GREEN + colorama.Style.BRIGHT}(automl) #> {colorama.Style.RESET_ALL}')
#         except KeyboardInterrupt:
#             break

#         agent.invoke(prompt)


# if __name__ == '__main__':
#     run_agent()
#     ...

# import gradio as gr
# import time

# def echo(message, history, system_prompt, tokens):
#     response = f"System prompt: {system_prompt}\n Message: {message}."
#     for i in range(min(len(response), int(tokens))):
#         time.sleep(0.05)
#         yield response[: i+1]

# with gr.Blocks() as demo:
#     system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")
#     slider = gr.Slider(10, 100, render=False)

#     gr.ChatInterface(
#         echo, additional_inputs=[system_prompt, slider],
#     )

# demo.launch()

import gradio as gr
from gradio import ChatMessage


agent = AutoMLAgent()


# def chat(message, history):
#     total = ""
#     for mode, value in agent.invoke(message):
#         if mode == 'messages' and value[1]["langgraph_node"] == "answer":
#             total += value[0].content
#             yield total

# ci = gr.ChatInterface(fn=chat, type="messages").launch()


def reset_agent():
    global agent, data_agent

    data_agent = DataAnalysisAgent()
    AutoMLAgent._modeling_agent = ModelingAgent()
    agent = AutoMLAgent()
    mempd.clear()

    print("RESET")


with gr.Blocks(fill_height=True) as demo:
    with gr.Row(scale=1):
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type="messages", scale=1)
            msg = gr.Textbox(scale=0, label="Nachricht")
            clear = gr.ClearButton([msg, chatbot], scale=0)
            clear.click(reset_agent)

            def respond(message, chat_history):
                def _parse_logs(_ls):
                    return "\n".join(_ls)
                
                _logs = []

                chat_history.append({"role": "user", "content": message})
                yield {
                    chatbot: chat_history,
                    cbg: gr.CheckboxGroup(choices=mempd.list()),
                    logs: _parse_logs(_logs)
                }

                step_message = ChatMessage(role="assistant", content="",
                                        metadata={"title": f"🛠️ Knotenprotokoll"})
                
                chat_history.append(step_message)

                steps = []

                total = ""
                datasets = []

                for graph, mode, value in agent.invoke(message):
                    if mode == 'custom' and value.get("answer_chunk"):
                        if not total:
                            chat_history.append({"role": "assistant", "content": ""})
                        chat_history[-1]["content"] += value["answer_chunk"]
                        total += value["answer_chunk"]
                        yield {
                            chatbot: chat_history,
                            cbg: gr.CheckboxGroup(choices=mempd.list()),
                            logs:  _parse_logs(_logs)
                        }
                    elif mode == "custom" and value.get("node_log"):
                        step_message.content += value["node_log"].replace(' ', '\t\t\t') + "\n"

                        yield {
                            chatbot: chat_history,
                            cbg: gr.CheckboxGroup(choices=mempd.list()),
                            logs:  _parse_logs(_logs)
                        }
                    elif mode == "custom" and value.get("alog"):
                        _logs.append(value["alog"])
                        yield {
                            chatbot: chat_history,
                            cbg: gr.CheckboxGroup(choices=mempd.list()),
                            logs: _parse_logs(_logs)
                        }       
                    elif mode == "custom" and value.get("new_dataset"):
                        datasets.append(value["new_dataset"])
                        yield {
                            chatbot: chat_history,
                            cbg: gr.CheckboxGroup(choices=mempd.list()),
                            logs: _parse_logs(_logs)
                        }

                chat_history[-1] = {"role": "assistant", "content": total}
                yield {
                    chatbot: chat_history,
                    cbg: gr.CheckboxGroup(choices=mempd.list()),
                    logs: _parse_logs(_logs)
                }
        
        with gr.Column(scale=1):
            cbg = gr.CheckboxGroup(label="Datensatzspeicher", scale=0)
            logs = gr.TextArea(label="Logs", scale=4)

            clear.add([logs, cbg])

    msg.submit(respond, [msg, chatbot], [chatbot, cbg, logs])

demo.launch()
