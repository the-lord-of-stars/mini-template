# Mini Challenge Template

Welcome! This repo provides: 

* `studio/`: A complete development playground for building and testing agent, and LangGraph Studio UI.
* `submission/`: The minimal structure you’ll package, compress in ZIP file, and submit (only what’s required to submit to evaluation server).



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
* OpenAI. Origianlly, the template uses OpenAI API. You can sign up [here](https://openai.com/index/openai-api/). 
    * Set `OPENAI_API_KEY` in your environment 

* Optional: LangSmith. Sign up for LangSmith [here](https://smith.langchain.com/). Use it within your workflow [here](https://www.langchain.com/langsmith), and relevant library [docs](https://docs.smith.langchain.com/).
    *    Set `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true` in your environment 

* Optional: Tavily. Tavily Search API is a web search engine optimized for LLMs and RAG, aimed at efficient, 
quick, and persistent search results. You can sign up for free for an API key [here](https://tavily.com/). 
    *   Set `TAVILY_API_KEY` in your environment. 

* Azure OpenAI. We provide a remote Azure LLM provider for free testing, and you can reach out to Pan Hao to get the AZURE_OPENAI_API_KEY and set the following environment variables:
```
export LLM_PROVIDER=azure
export AZURE_OPENAI_ENDPOINT=https://eval-models.openai.azure.com/
export AZURE_OPENAI_API_KEY=api_key
export AZURE_OPENAI_DEPLOYMENT=gpt-4o
```



## 🚀 Running the Studio Playground

### 1. Run template locally

```
python run.py
```

### 2. Run LangGraph Studio
LangGraph Studio is a custom IDE for viewing and testing agents, and it can be run locally and opened in your browser on Mac, Windows, and Linux.
See documentation about LangGraph CLI [here](https://langchain-ai.github.io/langgraph/cloud/reference/cli/) for both Python and JS configurations.

```
// from the studio folder
npx @langchain/langgraph-cli dev
```

* Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://localhost:2024`.
* Configuration file is the `/studio/langgraph.json` file.

### 3. Update your agentic workflow
Please ensure your agent configuration references `dataset.csv` as the data file, e.g., specifying in the prompt. Include every dependency in `requirements.txt`, confirm that `agent.py` defines your Agent class, and verify that `run.py` runs without errors before packaging your submission.


## 📬 Preparing Your Submission

1. Verify the code.

```
python run.py
```
2. Copy into the submission/ folder.
* `agent.py` - your Agent implementation 
* `requirements.txt` - all dependencies needed 
* `report.py` - the code to generate the report
* Other supplimentary files to run your agent, e.g., `helpers.py`

3. ZIP the submission/ folder (do not include any extra files or foler)
4. Submit the ZIP file via the challenge portal.



<!-- ### Optional: Set up Tavily API for web search

* Tavily Search API is a search engine optimized for LLMs and RAG, aimed at efficient, 
quick, and persistent search results. 
* You can sign up for an API key [here](https://tavily.com/). 
It's easy to sign up and offers a very generous free tier. Some lessons (in Module 4) will use Tavily. 

* Set `TAVILY_API_KEY` in your environment. -->

## References
- [Langchain Academy](https://github.com/langchain-ai/langchain-academy)