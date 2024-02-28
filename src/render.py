from jinja2 import Template
from datetime import datetime
from pathlib import Path

OUTPUT = "metadata.json"
TEMPLATE = "src/template.html"


import json

with open(OUTPUT, "r", encoding="UTF-8") as json_file:
    datasets = json.load(json_file)

# Define the content of the HTML page
title = "DLA"
heading = "🦑 Deep Lake Analytics"
paragraph = "Dive into your data lake with LLMs"

# Define the HTML template
template = Template(Path(TEMPLATE).read_text(encoding="UTF-8"))

runtime = -1
iso_date_time = datetime.now().astimezone().replace(microsecond=0).isoformat()

# Render the HTML template with the content
html_code = template.render(
    title=title,
    heading=heading,
    paragraph=paragraph,
    runtime=runtime,
    iso_date_time=iso_date_time,
    datasets=datasets,
)

# Write the HTML code to a file
with open("index.html", "w") as f:
    f.write(html_code)
