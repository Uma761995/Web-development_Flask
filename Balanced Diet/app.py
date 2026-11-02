
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request
from google import genai

# --------------------------------------------------
# 1. Configure paths and environment variables
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATHS = [
    BASE_DIR / "data" / "balanced_diet.csv",
    BASE_DIR / "data" / "organic_health_diet_foods.csv"
]

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
print("Loaded .env file:", BASE_DIR / ".env")
print("Gemini model selected:", MODEL_NAME)

client = genai.Client(api_key=API_KEY) if API_KEY else None

# --------------------------------------------------
# 2. Load the nutrition dataset
# --------------------------------------------------
def load_diet_data():
    datasets = []

    for file_path in DATA_PATHS:

        if not file_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {file_path}"
            )

        df = pd.read_csv(file_path)

        df.columns = df.columns.str.strip()

        # Remove completely empty rows
        df = df.dropna(how="all")

        if df.empty:
            raise ValueError(
                f"The CSV file is empty: {file_path.name}"
            )

        # Remember which dataset each row came from
        df["source_dataset"] = file_path.name

        datasets.append(df)

        print(
            f"Loaded {file_path.name}: "
            f"{df.shape[0]} rows, {df.shape[1]} columns"
        )

    return pd.concat(
        datasets,
        ignore_index=True,
        sort=False
    )

# --------------------------------------------------
# 3. Retrieve relevant CSV rows
# --------------------------------------------------

def retrieve_diet_context(question, df, limit=8):
    question_words = {
        word.lower().strip(".,?!:;()")
        for word in question.split()
        if len(word.strip(".,?!:;()")) > 2
    }

    # Compare the question with each row.
    searchable_df = df.fillna("").astype(str)

    scores = []

    for _, row in searchable_df.iterrows():
        row_text = " ".join(row.tolist()).lower()

        score = sum(
            1 for word in question_words
            if word in row_text
        )

        scores.append(score)

    # Return matching rows first.
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    matching_indices = [
        i for i in ranked_indices
        if scores[i] > 0
    ][:limit]

    # If nothing matches, provide a small sample of the dataset.
    if not matching_indices:
        matching_indices = list(range(min(limit, len(df))))

    context_df = df.iloc[matching_indices]

    return context_df.to_csv(index=False)

# --------------------------------------------------
# 4. Generate an answer with Gemini
# --------------------------------------------------

def generate_diet_answer(question, diet_context):
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add your key to the .env file."
        )

    prompt = f"""
You are a friendly Balanced Diet Chatbot.

Your responsibilities:
- Answer questions about balanced meals, breakfast, lunch,
  dinner, fruits, nuts, millets, and general nutrition.
- Use the provided CSV records when they are relevant.
- Do not invent food records or nutrition figures.
- If a requested food is not present in the dataset,
  clearly say that the dataset does not contain that information.
- Explain food suggestions in simple, easy-to-understand English.
- For diabetes, blood pressure, and thyroid questions, give
  general educational information only.
- Never diagnose diseases, recommend medication changes,
  or claim that a food cures a disease.
- Mention that individual dietary needs vary and recommend
  professional advice for personalized medical nutrition.
- If the question is unrelated to nutrition, politely guide
  the user back to the chatbot's intended topics.

CSV DATA:
{diet_context}

USER QUESTION:
{question}

Write a helpful, concise answer.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    answer = response.text

    if not answer:
        return (
            "Gemini did not return a text answer. "
            "Please try asking your question again."
        )

    return answer.strip()

# --------------------------------------------------
# 5. Flask route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    question = ""
    answer = ""
    error = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if not question:
            error = "Please enter a question."

        elif len(question) > 1000:
            error = "Please keep your question under 1000 characters."

        else:
            try:
                df = load_diet_data()

                context = retrieve_diet_context(
                            question,
                            df,
                            limit=8
                            )

                answer = generate_diet_answer(
                    question,
                    context
                )

            except FileNotFoundError:
                app.logger.exception("CSV file missing")
                error = (
                    "Diet dataset not found. Check "
                    "data/balanced_diet.csv."
                )

            except pd.errors.ParserError:
                app.logger.exception("CSV parsing failed")
                error = (
                    "Unable to read the CSV. Check its format "
                    "and column separators."
                )

            except Exception:
                app.logger.exception("Chatbot request failed")
                error = (
                    "Sorry, the chatbot could not generate a response. "
                    "Check your API key, model access, internet connection, "
                    "and terminal error message."
                )

    return render_template(
        "index.html",
        question=question,
        answer=answer,
        error=error
    )

# --------------------------------------------------
# 6. Start the application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )