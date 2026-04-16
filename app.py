import re
import io
import base64

from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# ========== LLM: Qwen2-0.5B-Instruct ==========
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cpu")


def extract_section(text, title):
    pattern = rf"{title}(.*?)(?=\n[A-Z][A-Za-z \-]+?:|\Z)"
    match = re.search(pattern, text, re.S)
    return match.group(1).strip() if match else ""


def parse_table(section, n_cols):
    lines = [l for l in section.splitlines() if l.strip()]
    lines = [l for l in lines if not set(l.strip()) <= set("_-")]
    data = []
    for line in lines:
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) >= n_cols:
            data.append(parts[:n_cols])
    return data


def df_from_sections(text):
    tally_raw = extract_section(text, "TALLY VARIABLES")
    discrete_raw = extract_section(text, "DISCRETE-CHANGE VARIABLES")
    outputs_raw = extract_section(text, "OUTPUTS")

    df_tally = pd.DataFrame(
        parse_table(tally_raw, 6),
        columns=["Identifier", "Average", "HalfWidth", "Minimum", "Maximum", "Observations"]
    )
    df_discrete = pd.DataFrame(
        parse_table(discrete_raw, 6),
        columns=["Identifier", "Average", "HalfWidth", "Minimum", "Maximum", "FinalValue"]
    )
    df_outputs = pd.DataFrame(
        parse_table(outputs_raw, 2),
        columns=["Identifier", "Value"]
    )

    for df in [df_tally, df_discrete, df_outputs]:
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(
                df[col].astype(str)
                .str.replace("Insuf", "", regex=False)
                .str.replace("(", "", regex=False)
                .str.replace(")", "", regex=False)
                .str.strip(),
                errors="coerce"
            )

    return df_tally, df_discrete, df_outputs, tally_raw, discrete_raw, outputs_raw


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


def generar_graficas(df_tally, df_discrete):
    # Tiempos de espera
    mask_wait = df_tally["Identifier"].str.contains("WaitTime", case=False, na=False)
    df_wait = df_tally[mask_wait].copy()

    wait_img = None
    if not df_wait.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(df_wait["Identifier"], df_wait["Average"])
        ax.set_xlabel("Tiempo promedio de espera")
        ax.set_title("Tiempos de espera en colas")
        ax.invert_yaxis()
        wait_img = fig_to_base64(fig)

    # Utilización
    mask_util = df_discrete["Identifier"].str.contains("Utilization", case=False, na=False)
    df_util = df_discrete[mask_util].copy()

    util_img = None
    if not df_util.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df_util["Identifier"], df_util["Average"])
        ax.set_ylabel("Utilización promedio")
        ax.set_title("Utilización de recursos")
        plt.xticks(rotation=45, ha="right")
        util_img = fig_to_base64(fig)

    # WIP
    mask_wip = df_discrete["Identifier"].str.contains("WIP", case=False, na=False)
    df_wip = df_discrete[mask_wip].copy()

    wip_img = None
    if not df_wip.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df_wip["Identifier"], df_wip["Average"])
        ax.set_ylabel("WIP promedio")
        ax.set_title("Work In Process (WIP)")
        plt.xticks(rotation=45, ha="right")
        wip_img = fig_to_base64(fig)

    return wait_img, util_img, wip_img


def generar_resumen_llm(tally_raw, discrete_raw, outputs_raw):
    prompt = f"""
Eres un experto en simulación discreta y análisis de sistemas.
Genera un resumen breve (máx. 10 líneas) con:
- Qué muestra la simulación
- Principales problemas
- 3 propuestas de mejora concretas

--- TALLY VARIABLES ---
{tally_raw}

--- DISCRETE VARIABLES ---
{discrete_raw}

--- OUTPUTS ---
{outputs_raw}
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.4,
            top_p=0.9
        )

    texto = tokenizer.decode(output[0], skip_special_tokens=True)
    # Opcional: recortar el prompt del inicio si el modelo lo repite
    if prompt.strip() in texto:
        texto = texto.split(prompt.strip(), 1)[-1].strip()
    return texto


@app.route("/", methods=["GET", "POST"])
def index():
    resumen = None
    wait_img = util_img = wip_img = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".out"):
            content = file.read().decode("utf-8", errors="ignore")

            df_tally, df_discrete, df_outputs, tally_raw, discrete_raw, outputs_raw = df_from_sections(content)
            wait_img, util_img, wip_img = generar_graficas(df_tally, df_discrete)
            resumen = generar_resumen_llm(tally_raw, discrete_raw, outputs_raw)

    return render_template(
        "index.html",
        resumen=resumen,
        wait_img=wait_img,
        util_img=util_img,
        wip_img=wip_img
    )


if __name__ == "__main__":
    app.run(debug=True)
