import re
import nbformat
import subprocess

with open('Actionable_Insights.py', 'r') as f:
    text = f.read()

# Remove any existing Jupytext markers to start fresh
text = re.sub(r'# %%.*\n', '', text)

# Replace all # In[...]: with # %%
text = re.sub(r'# In\[\d+\]:', '# %%', text)

# Add # %% [markdown] before the very first header
text = re.sub(r'(# # Actionable Insights)', r'# %% [markdown]\n\1', text)

# Add # %% [markdown] before any section header (starts with # ## or # ###)
# But wait, there might be text under it.
text = re.sub(r'\n(# ## .*?\n)', r'\n# %% [markdown]\n\1', text)
text = re.sub(r'\n(# ### .*?\n)', r'\n# %% [markdown]\n\1', text)

# Add # %% [markdown] before any Insight block
text = re.sub(r'\n(# \*\*Insight\*\*:.*?\n)', r'\n# %% [markdown]\n\1', text)

# Write modified text back
with open('Actionable_Insights_fixed.py', 'w') as f:
    f.write(text)

print("Formatting fixed!")
