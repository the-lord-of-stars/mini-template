# Mini Challenge Template

Welcome! This repo provides: 

* `studio/`: A complete development playground for building and testing agent with LangGraph Studio UI. Please run the template locally in this folder. 
* `submission/`: The minimal structure you’ll package, compress in ZIP file, and submit (only what’s required to submit to evaluation server).

Please follow the instructions below to get started.
- [0 - Prerequisites](#0-prerequisites)
- [1 - Configure local API](#1-configure-api)
- [2 - Begin the Template and Studio](#2-begin-the-template-and-studio)
- [3 - Preparing Your Submission](#3-preparing-your-submission)

## 0 Prerequisites

- Python 3.11+

This version is required for optimal compatibility with LangGraph. If you're on an older version, upgrading will ensure everything runs smoothly.
```
python --version
```
```
git clone https://github.com/ppphhhleo/mini-template.git
$ cd mini-template
```

### Create an environment and install dependencies
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

## 1 Configure API 
* OpenAI. Origianlly, the template uses OpenAI API to run **locally**. You can sign up [here](https://openai.com/index/openai-api/). 
    * Set `OPENAI_API_KEY` in `.studio/.env` file,
    * Set `OPENAI_MODEL = "gpt-4o"` in `.studio/.env` file,
    * Set `LLM_PROVIDER = "openai"` in `.studio/.env` file
    The evaluation server uses Azure OpenAI to run your submission once you upload your submission, and you do *not* need to fill in the Azure OpenAI endpoint, API key, and deployment name.

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


## 2 Begin the Template and Studio

### 2.1 Run the template 

```
cd studio
python run.py
```

If the template is running successfully the first time, the output will be saved in the `studio/output.pdf` file. 

If you want the template to generate Vega-Lite charts, please adjust the prompt and decode_output function in the `agent.py` file accordingly (see the comments in the `agent.py` file), then run the following command to view the output in the browser `http://localhost:8001/output.html`.

```
python run.py
python -m http.server 8001
```


### 2.2 Optionally, Run LangGraph Studio 
LangGraph Studio is a custom IDE for viewing and testing agents, and it can be run locally and opened in your browser on Mac, Windows, and Linux.
See documentation about LangGraph CLI [here](https://langchain-ai.github.io/langgraph/cloud/reference/cli/) for both Python and JS configurations.

```
npx @langchain/langgraph-cli dev
```
* Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://localhost:2024`.
* Configuration file is the `/studio/langgraph.json` file.

### 2.3 Update your agentic configuration

Feel free to customize the `agent.py` and its companion files to craft your own agentic configuration, and try to make your designs and results outperform the template baseline, by demonstrating your solution's **generalizability to other datasets**, **efficiency in running**, and **effectiveness in generating effective and engaging narrative-driven visualization reports**.

Please ensure you refer to `dataset.csv` as the data file as input, e.g., specifying in the prompt; and refer to `output.html` or `output.pdf` as the output file, and include every dependency needed in `requirements.txt`


## 3 Preparing Your Submission

1. Verify the codes. Please make sure the codes execute without errors before packing your submission.

```
python run.py
```


2. Copy related files into the `submission/` folder. Currently submission folder contains example files to generate PDF report, please replace them with your own files.
* `agent.py` - your Agent implementation (Required)
* `requirements.txt` - all dependencies needed  (Required)
* All supplimentary files if any, e.g., `helpers.py`, `report_html.py`, `report_pdf.py`, etc.

3. ZIP the `submission/` folder (do not include any extra files or foler)
4. Submit the ZIP file via the [challenge website](https://purple-glacier-014f19d1e.6.azurestaticapps.net/) to see the result, and you could submit multiple times.
5. Submit your paper via PCS. 


## References
- [Langchain Academy](https://github.com/langchain-ai/langchain-academy)