# gds-idea-ragchat-agent


A repo to hold the agent from our ragchat app. 
The longterm goal is to opensource the ragchat application. 
To enable that we have separated the agent and infrastructure code.
This will allow us to more rapidly make changes to the workings of the agent.


# Running the agent locally

The project uses `uv` to manage dependencies.
We recommned installing `uv` via homebrew on macs.  

```bash
git clone co-cddo/gds-idea-ragchat-agent.git
cd gds-idea-ragchat-agent
uv sync
```

See `main.py` for an example of how to call the agent. 