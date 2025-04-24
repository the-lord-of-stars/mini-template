# mini-template

## Introduction

Welcome to Mini Challenge.
In `studio`  folder, you'll see example codes to build multi-agent workflows using LangGraph. 

## Setup

### Python version

To get the most out of this course, please ensure you're using Python 3.11 or later. 
This version is required for optimal compatibility with LangGraph. If you're on an older version, 
upgrading will ensure everything runs smoothly.
```
python --version
```

### Clone repo
```
git clone https://github.com/ppphhhleo/mini-template.git
$ cd mini-template
```

### Create an environment and install dependencies
#### Mac/Linux/WSL
```
$ python -m venv mini-template-env
$ source mini-template-env/bin/activate
$ pip install -r requirements.txt
```
#### Windows Powershell
```
PS> python -m venv mini-template-env
PS> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
PS> mini-template-env\scripts\activate
PS> pip install -r requirements.txt
```

### Set OpenAI API key
* If you don't have an OpenAI API key, you can sign up [here](https://openai.com/index/openai-api/).
*  Set `OPENAI_API_KEY` in your environment 

### Sign up LangSmith 
* Sign up for LangSmith [here](https://smith.langchain.com/), find out more about LangSmith


### Set up LangGraph Studio

* LangGraph Studio is a custom IDE for viewing and testing agents, and it can be run locally and opened in your browser on Mac, Windows, and Linux.
<!-- * See documentation [here](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/#local-development-server) on the local Studio development server and [here](https://langchain-ai.github.io/langgraph/how-tos/local-studio/#run-the-development-server).  -->
* See documentation about LangGraph CLI [here](https://langchain-ai.github.io/langgraph/cloud/reference/cli/) for both Python and JS configurations.
* Graphs and codes for LangGraph Studio are in the `/studio` folders.
* To start the local development server, run the following command in your terminal in the `/studio` directory:

```
npx @langchain/langgraph-cli dev
```

Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.

### Optional: Set LangSmith API
* Use it within your workflow [here](https://www.langchain.com/langsmith), and relevant library [docs](https://docs.smith.langchain.com/)!
* Set `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true` in your environment 

### Optional: Set up Tavily API for web search

* Tavily Search API is a search engine optimized for LLMs and RAG, aimed at efficient, 
quick, and persistent search results. 
* You can sign up for an API key [here](https://tavily.com/). 
It's easy to sign up and offers a very generous free tier. Some lessons (in Module 4) will use Tavily. 

* Set `TAVILY_API_KEY` in your environment.

## References
- [Langchain Academy](https://github.com/langchain-ai/langchain-academy)