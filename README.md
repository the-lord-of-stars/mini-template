# Mini Challenge Template

Welcome! This repo provides: 

* **studio/**: A complete development playground for building and testing agent, and LangGraph Studio UI.
* **submission/**: The minimal structure you’ll package, compress in ZIP file, and submit (only what’s required to submit to evaluation server).



## Quick Start

### Python version

- Python 3.11+

This version is required for optimal compatibility with LangGraph. If you're on an older version, upgrading will ensure everything runs smoothly.
```
python --version
```

### 1. Clone repo
```
git clone https://github.com/ppphhhleo/mini-template.git
$ cd mini-template/studio
```

### 2. Create an environment and install dependencies
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

### 3. Configure API 
* OpenAI. If you don't have an OpenAI API key, you can sign up [here](https://openai.com/index/openai-api/). 
    * Set `OPENAI_API_KEY` in your environment 

* Optional: LangSmith. Sign up for LangSmith [here](https://smith.langchain.com/). Use it within your workflow [here](https://www.langchain.com/langsmith), and relevant library [docs](https://docs.smith.langchain.com/)!
    *    Set `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true` in your environment 



## 🚀 Running the Studio Playground

### 1. Run template locally

```
python run.py
```

### 2. Run LangGraph Studio
LangGraph Studio is a custom IDE for viewing and testing agents, and it can be run locally and opened in your browser on Mac, Windows, and Linux.
See documentation about LangGraph CLI [here](https://langchain-ai.github.io/langgraph/cloud/reference/cli/) for both Python and JS configurations.

```
npx @langchain/langgraph-cli dev
```

* Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://localhost:2024`.
* Configuration file is the `/studio/langgraph.json` file.




<!-- ### Optional: Set up Tavily API for web search

* Tavily Search API is a search engine optimized for LLMs and RAG, aimed at efficient, 
quick, and persistent search results. 
* You can sign up for an API key [here](https://tavily.com/). 
It's easy to sign up and offers a very generous free tier. Some lessons (in Module 4) will use Tavily. 

* Set `TAVILY_API_KEY` in your environment. -->

## References
- [Langchain Academy](https://github.com/langchain-ai/langchain-academy)