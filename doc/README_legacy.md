A draft repository to start evaluating AI applications at Data Gouv.

## Set up an evaluation platform

To evaluate : 
- the MCP server
- later, any AI use case like the LLM integration in the features of the website

### Platforms benchmark

- **Phoenix Arize** \[Tested]. Biggest limitation : not possible to calculate aggregated metrics over a dataset (only per "example" metrics)
- **MLFlow GenAI**. Not an LLM-native framework, comes from classic ML, but might be robust and provide both an integration to other framework like DeepEval/Phoenix, AND a UI for them. Biggest limitation : because they just started their journey into the GenAI world, many important features still seem to be available only on DataBricks.
- **DeepEval**. Probably one of the most known. Used by the IAE. Biggest limitation : The open source framework has no self-hosted UI. The free cloud UI is only for one project, two users.
- **ZenML**. It's a whole platform not only for evaluation but also training. It looks like more of an open-source Dataiku. Biggest limitation: because it covers the whole lifecycle of an ML project and both GenAI + classic ML, it has core concepts to learn, too many features... It seems overkill for our usage.
- **Opik** \[Next test]. Less popular as newcomer, but promising. Open source UI integrated. They provided many metrics, but also datasets agregated metrics which is much needed. To test !
- many others exist but hard to benchmark all : RAGAS seems too RAG-oriented, heard that LangFuse were incomplete when it comes to evaluation (best for tracing), ... 

## First evaluation: the MCP server.

### Context

An MCP server shouldn't be only tested from an infra point of view, as it is meant to be part of an agentic workflow.
Agents take decision based on their understanding of the MCP and its tools, prompts, resources. 
It means that like any variable in an agentic workflow, we should test the interactions between the LLM and the MCP information we give to it : 
- tools name
- arguments name
- descriptions
- and more like tools quantity, overlapping functions, etc...

We not only need to evaluate the overall performance but the added cost (in terms of token, iterations which in the end translate to money + energy consumption).
Does the MCP overload the agents context because of too many tools ? too long description ? can we make the descriptions shorter to avoid token unecessary use? 
or are the name not obvious enough that agents get confused and run too many iteration - or even doesn't find the right answer ? 
Many questions that can be answered through rigorous evaluations.

### Goals:
- evaluate the ability of the MCP to enhance the discoverability of the available datasets and APIs and their content
- better understand the agentic usage of the MCP and possibly extend the MPC offer (for now tools only - later, maybe add prompts, ressources...) 
- compare version of the MCP to improve it
- ensure the MCP performance doesn't regress when pushing new versions : integration in the CI/CD some eval checks
- while it's just the beginning of our MCP journey, because of the buzz around it and the possible future usage, it's better to demonstrate its robustness and what we do about it

To ensure both result robustness while being efficient, the evaluations should be run in priority :
- over a selection of models and providers : we can select top 3 + some from Albert API as users of the AI assistant might be power users
- over prompts based on the queries from the top X of the queries in the search from data gouv AND from identified users use cases
- test different levels of complexity
- test the different tools

Metrics should not only include performance metrics like tool selection and invocation correctness, but also token cost, latency, etc.



