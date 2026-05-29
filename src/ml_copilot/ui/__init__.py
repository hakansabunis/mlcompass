"""Rich-based terminal rendering for ml-copilot output.

Each module here renders the structured output of one agent or tool
into a human-friendly terminal layout. Keeping rendering separate from
agent logic lets us swap UIs (rich → JSON → JupyterDisplay) without
touching the agents.
"""
