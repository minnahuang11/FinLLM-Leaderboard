import os
import gradio as gr

# Read the SVG content
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level and then into the Resource folder.
svg_path = os.path.join(current_dir, "..", "Resource", "search-outline.svg")
with open(svg_path, "r") as file:
    svg_content = file.read()

custom_css = """
/* Container that holds the button + tooltip */
.hover-container {
    position: relative;
    display: inline-block;
    margin: 1rem;
    overflow: visible;
}

/* The tooltip, initially hidden */
.hover-tooltip {
    visibility: hidden;
    opacity: 0;
    transition: opacity 0.3s ease;
    width: 240px;
    background-color: #333;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 999;
}

/* Tooltip arrow pointing upward */
.hover-tooltip::after {
    content: "";
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    border-width: 5px;
    border-style: solid;
    border-color: transparent transparent #333 transparent;
}

/* Show the tooltip on hover */
.hover-container:hover .hover-tooltip {
    visibility: visible;
    opacity: 1;
}

/* Override overflow for Gradio’s html container */
.html-container {
    overflow: visible !important;
    padding: 0 !important;
}

/* Override overflow for additional Gradio containers */
.hide-container,
.block {
    overflow: visible !important;
}

/* Disable padding for svelte padded container */
.padded.svelte-11xb1hd {
    padding: 0 !important;
}

/* Add padding for the container with id "component-2" */
#component-2 {
    padding: 10px !important;
    border: 1px solid #ccc; 
}

#component-3 {
    padding-right: 0 !important;
}

/* Remove extra space for the SVG icon container */
.svg-icon-container {
    margin: 0 !important;
    padding: 0 !important;
    display: inline-block;
}

/* Constrain SVG icon size */
.svg-icon-container svg {
    height: 20px;
    width: auto;
    display: inline-block;
    vertical-align: middle;
}
"""

models_data = [
    {
        "Rank": 1,
        "Type": "Type A",
        "Model": "Maziyar/PaLM2Alpha",
        "Average": 60.12,
        "IFEval": 70.5,
        "BBH": 65.3,
        "MATH": 55.2,
        "GPQA": 60.1,
        "MUSR": 58.4,
        "MMLU-PRO": 62.3,
        "CO2 Cost": 1.2
    },
    {
        "Rank": 2,
        "Type": "Type B",
        "Model": "Maziyar/PantheraLlama",
        "Average": 59.85,
        "IFEval": 69.8,
        "BBH": 64.7,
        "MATH": 54.9,
        "GPQA": 59.7,
        "MUSR": 57.9,
        "MMLU-PRO": 61.8,
        "CO2 Cost": 1.3
    },
    {
        "Rank": 3,
        "Type": "Type C",
        "Model": "open-llm/falcon-7b",
        "Average": 59.73,
        "IFEval": 69.5,
        "BBH": 64.3,
        "MATH": 54.5,
        "GPQA": 59.3,
        "MUSR": 57.5,
        "MMLU-PRO": 61.3,
        "CO2 Cost": 1.4
    },
]

def filter_models(search_query, min_score, max_score):
    filtered = []
    for item in models_data:
        if (search_query.lower() in item["Model"].lower()) and (min_score <= item["Average"] <= max_score):
            filtered.append(item)
    return filtered

def display_leaderboard(search_query, min_score, max_score):
    filtered = filter_models(search_query, min_score, max_score)
    return [
        [m["Rank"], m["Type"], m["Model"], m["Average"], m["IFEval"], m["BBH"], m["MATH"], m["GPQA"], m["MUSR"], m["MMLU-PRO"], m["CO2 Cost"]]
        for m in filtered
    ]

# Prepare initial data to display when the app loads
initial_data = [
    [m["Rank"], m["Type"], m["Model"], m["Average"], m["IFEval"], m["BBH"], m["MATH"], m["GPQA"], m["MUSR"], m["MMLU-PRO"], m["CO2 Cost"]]
    for m in models_data
]

with gr.Blocks(css=custom_css) as demo:
    
    gr.Markdown(
        """
        <h1 style="text-align:center;">Open FIN LLM Leaderboard</h1>
        <p style="text-align:center;">
            Comparing Large Language Models in an open and reproducible way
        </p>
        """
    )
    
    with gr.Row(elem_classes="no-clip"):
        gr.HTML(f'<div class="svg-icon-container">{svg_content}</div>')
        search_box = gr.Textbox(placeholder="Search by model name", show_label=False)
        AdvancedSearchBtn = gr.Button("Advanced Filters")
        
        tooltip_html = """
                <div class="hover-container">
                    <!-- This is the element that triggers the tooltip on hover -->
                    <button style="cursor: pointer;">ⓘ</button>
                    
                    <!-- The tooltip content -->
                    <div class="hover-tooltip">
                        <strong>Name Search:</strong> Search directly by model name<br>
                        <strong>Field Search:</strong> Use field:value syntax<br>
                        <strong>Multiple Searches:</strong> Combine multiple criteria
                    </div>
                </div>
                """
        gr.HTML(tooltip_html)
    
    with gr.Group():
        results_table = gr.Dataframe(
            headers=["Rank", "Type", "Model", "Average", "IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU-PRO", "CO2 Cost"],
            datatype=["number", "str", "str", "number", "number", "number", "number", "number", "number", "number", "number"],
            interactive=False,
            value=initial_data
        )

demo.launch()