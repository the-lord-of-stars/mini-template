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
$ cd mini-template
```

### 2. Create an environment and install dependencies
#### Mac/Linux/WSL
```
$ python -m venv mini-template-env
$ source mini-template-env/bin/activate
$ cd mini-template/studio
$ pip install -r requirements.txt
```
#### Windows Powershell
```
PS> python -m venv mini-template-env
PS> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
PS> mini-template-env\scripts\activate
PS> cd mini-template/studio
PS> pip install -r requirements.txt
```

### 3. Configure API 
* OpenAI. Origianlly, the template uses OpenAI API to run locally. You can sign up [here](https://openai.com/index/openai-api/). 
    * Set `OPENAI_API_KEY` in `.studio/.env` file,
    * Set `OPENAI_MODEL = "gpt-4o"` in `.studio/.env` file,
    * Set `LLM_PROVIDER = "openai"` in `.studio/.env` file
    The evaluation server uses Azure OpenAI to run your submission once you upload your submission, and you do not need to fill in the Azure OpenAI endpoint, API key, and deployment name.

* [Optional]: LangSmith. Sign up for LangSmith [here](https://smith.langchain.com/). Use it within your workflow [here](https://www.langchain.com/langsmith), and relevant library [docs](https://docs.smith.langchain.com/).
    *    Set `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true` in `.studio/.env` file 

* [Optional]: Tavily. Tavily Search API is a web search engine optimized for LLMs and RAG, aimed at efficient, 
quick, and persistent search results. You can sign up for free for an API key [here](https://tavily.com/). 
    *   Set `TAVILY_API_KEY` in `.studio/.env` file. 

<!-- * Azure OpenAI. We provide a remote Azure LLM provider for free testing, and you can reach out to Pan Hao to get the AZURE_OPENAI_API_KEY and set the following environment variables:
```
export LLM_PROVIDER=azure
export AZURE_OPENAI_ENDPOINT=https://eval-models.openai.azure.com/
export AZURE_OPENAI_API_KEY=api_key
export AZURE_OPENAI_DEPLOYMENT=gpt-4o
``` -->


## 🚀 Begin with the Template & Studio

### 1. Run template locally

```
cd studio
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

### 3. Update your agentic workflow
You can update the `agent.py` and related files to implement your own agentic configuration, and try to make the design and results better than the template baseline. 

Please ensure you refer to `dataset.csv` as the data file, e.g., specifying in the prompt; and include every dependency in `requirements.txt`


## 📬 Preparing Your Submission

1. Verify the codes. Please make sure the codes execute without errors before packing your submission.

```
python run.py
```


2. Copy related files into the `submission/` folder.
* `agent.py` - your Agent implementation 
* `requirements.txt` - all dependencies needed 
* `report.py` - the code to generate the report
* all other supplimentary files to run your agent, e.g., `helpers.py`, `report_html.py`, `report_pdf.py`, etc.

3. ZIP the `submission/` folder (do not include any extra files or foler)
4. Submit the ZIP file via the [challenge website](https://purple-glacier-014f19d1e.6.azurestaticapps.net/) to see the result, and you could submit multiple times.
5. Submit your paper via PCS. 


## References
- [Langchain Academy](https://github.com/langchain-ai/langchain-academy)