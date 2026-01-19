from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file, flash
import plotly
from plotly.offline import plot
import plotly.graph_objs as go
import pandas as pd
from pandas import Period

import openai
import json
import os

app = Flask(__name__)
app.secret_key = 'secret-key-for-session'

raw_uploaded_df = None
BUDGETS_FILE = 'budgets.json'

def get_api_key():
    try:
        with open("config.json") as f:
            config = json.load(f)
        return config.get("OPENAI_API_KEY", "")
    except FileNotFoundError:
        return ""

openai.api_key = get_api_key()

# ---------- Budget Persistence ----------
def load_budgets():
    if os.path.exists(BUDGETS_FILE):
        with open(BUDGETS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_budgets(budgets):
    with open(BUDGETS_FILE, 'w') as f:
        json.dump(budgets, f, indent=2)

def load_transactions_for_month(selected_month):
    df = pd.read_csv(f'processed_{selected_month}.csv')
    return df.to_dict(orient='records')

def load_budgets_for_month(selected_month):
    budgets = load_budgets()
    return budgets.get(selected_month, {})

def calculate_spending(selected_month):
    df = pd.read_csv(f'processed_{selected_month}.csv')
    return df.groupby('Category')['Amount'].sum().to_dict()

def query_llm(prompt_text):
    try:
        client = openai.OpenAI(api_key=get_api_key())
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies expenses into predefined categories."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error:\n{e}")
        return "Other"


def generate_budget_recommendation(budget_data, actual_spending_data):
    prompt = "Based on the following category budgets and actual spendings, suggest new budget amounts for next month. Increase budgets for categories with consistent overspending and decrease budgets for underutilized categories, but keep changes moderate.\n\n"
    for category in budget_data:
        budgeted = budget_data[category]
        spent = actual_spending_data.get(category, 0)
        prompt += f"- {category}: Budgeted ${budgeted}, Spent ${spent}\n"

    prompt += "\nPlease return the new suggested budget for each category in the format:\nCategory: $Amount\nExample:\nGroceries: $250\nDining Out: $80\nTransport: $70"

    client = openai.OpenAI(api_key=get_api_key())
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content

@app.route('/accept_recommendations', methods=['POST'])
def accept_recommendations():
    try:
        selected_month = session.get('selected_month')
        if not selected_month:
            flash("No month selected.", "error")
            return redirect(url_for('dashboard'))

        recommendations_text = request.form.get('recommendations_text')
        if not recommendations_text:
            flash("No recommendations found.", "error")
            return redirect(url_for('dashboard'))

        new_budgets = {}
        lines = recommendations_text.strip().split('\n')
        for line in lines:
            if ':' in line:
                category, amount = line.split(':', 1)
                category = category.strip()
                amount = amount.strip().replace('$', '')
                try:
                    amount_value = float(amount)
                    new_budgets[category] = amount_value
                except ValueError:
                    continue  # skip lines that can't parse

        from pandas import Period

        current_month = Period(selected_month, freq='M')
        next_month = (current_month + 1).strftime('%Y-%m')

        budgets = load_budgets()
        budgets[next_month] = new_budgets
        save_budgets(budgets)

        session['selected_month'] = next_month

        flash(f"New recommended budgets saved for {next_month}!", "success")
        return redirect(url_for('dashboard'))  
    except Exception as e:
        flash(f"Error saving new budgets: {str(e)}", "error")
        return redirect(url_for('dashboard'))


def clean_response_lines(response, expected_count):
    valid_categories = {"Food", "Transportation", "Entertainment", "Shopping", "Other"}
    if not response:
        return ["Other"] * expected_count

    raw_lines = response.split(",") if "," in response and "\n" not in response else response.split("\n")
    cleaned = []
    for line in raw_lines:
        line = line.strip().strip(".").split(".")[-1].split(":")[-1].strip()
        cleaned.append(line if line in valid_categories else "Other")

    if len(cleaned) != expected_count:
        return ["Other"] * expected_count

    return cleaned

def categorize_expenses_batch(descriptions, batch_size=5):
    all_categories = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        prompt = (
            f"You will be given a list of {len(batch)} expense descriptions. "
            "For each one, respond with exactly one of the following categories: "
            "Food, Transportation, Entertainment, Shopping, Other.\n\n"
            "Respond with only the category names, one per line, in the same order.\n\n"
            "Expenses:\n" +
            "\n".join(batch) +
            "\n\nResponse:"
        )

        response = query_llm(prompt)
        categories = clean_response_lines(response, len(batch))

        if categories.count("Other") == len(batch):
            retry_response = query_llm(prompt)
            categories = clean_response_lines(retry_response, len(batch))

        if categories.count("Other") == len(batch):
            categories = []
            for desc in batch:
                single_response = query_llm(desc)
                category = clean_response_lines(single_response, 1)[0]
                categories.append(category)

        all_categories.extend(categories)
    return all_categories

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/process_upload', methods=['POST'])
def process_upload():
    global raw_uploaded_df

    file = request.files['file']
    if not file:
        return "No file uploaded.", 400

    df = pd.read_csv(file)
    if "Description" not in df.columns or "Amount" not in df.columns or "Date" not in df.columns:
        return "CSV must have 'Description', 'Amount', and 'Date' columns.", 400

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    raw_uploaded_df = df

    months = sorted(df['Month'].unique())
    return render_template('select_month.html', months=months)

@app.route('/filter_month', methods=['POST'])
def filter_month():
    global raw_uploaded_df
    selected_month = request.form.get("month")
    session['selected_month'] = selected_month

    if raw_uploaded_df is None or not selected_month:
        return "No data or month selected.", 400

    df = raw_uploaded_df[raw_uploaded_df['Month'] == selected_month].copy()
    df['Category'] = categorize_expenses_batch(df['Description'].astype(str).tolist())
    df.to_csv(f'processed_{selected_month}.csv', index=False)

    return redirect(url_for('budget_page'))

@app.route('/budget')
def budget_page():
    return render_template('budget.html')

@app.route('/set_budget', methods=['POST'])
def set_budget():
    selected_month = session.get('selected_month')
    if not selected_month:
        return "Month not selected", 400

    budgets = load_budgets()
    budgets[selected_month] = {
        "Food": float(request.form.get('food', 0.0)),
        "Transportation": float(request.form.get('transportation', 0.0)),
        "Entertainment": float(request.form.get('entertainment', 0.0)),
        "Shopping": float(request.form.get('shopping', 0.0)),
        "Other": float(request.form.get('other', 0.0))
    }
    save_budgets(budgets)
    return redirect(url_for('insights_page'))

@app.route('/recommend_budgets')
def recommend_budgets():
    selected_month = session.get('selected_month')
    if not selected_month:
        flash('Please select a month first.', 'error')
        return redirect(url_for('upload_page'))

    try:
        transactions = load_transactions_for_month(selected_month)
        budgets = load_budgets_for_month(selected_month)

        if not budgets:
            flash('Please set budgets first for this month.', 'error')
            return redirect(url_for('budget_page'))

        actual_spending = {}
        for t in transactions:
            category = t['Category']
            amount = float(t['Amount'])
            actual_spending[category] = actual_spending.get(category, 0) + amount

        prompt = (
            "You are a budgeting expert helping users adjust their monthly budgets based on their real spending.\n\n"
            f"For the month of {selected_month}, here are the stats:\n\n"
        )
        for category in budgets:
            prompt += f"- {category}: Budgeted ${budgets[category]:.2f}, Spent ${actual_spending.get(category, 0):.2f}\n"

        prompt += (
            "\nSuggest new budgets for next month.\n"
            "Increase the budget slightly for consistently overspent categories.\n"
            "Decrease budgets slightly for underutilized categories.\n"
            "Keep suggestions moderate and balanced.\n\n"
            "Format your response as:\n"
            "Category: $NewAmount"
        )

        recommendation_text = query_llm(prompt)

        return render_template('recommendations.html', recommendations=recommendation_text)

    except Exception as e:
        flash(f"Error generating recommendations: {str(e)}", 'error')
        return redirect(url_for('dashboard'))



@app.route('/transactions')
def transactions_page():
    selected_month = session.get('selected_month')
    if not selected_month:
        flash('Please select a month.', 'error')
        return redirect(url_for('upload_page'))

    try:
        df = pd.read_csv(f'processed_{selected_month}.csv')
        spending = df.groupby('Category')['Amount'].sum().to_dict()

        return render_template('transactions.html',
            transactions=df.to_dict(orient='records'),
            spending=spending
        )

    except FileNotFoundError:
        flash('Transactions file not found for selected month.', 'error')
        return redirect(url_for('upload_page'))
    except Exception as e:
        return f"Transactions error: {str(e)}"


@app.route('/insights')
def insights_page():
    selected_month = session.get('selected_month')
    if not selected_month:
        return "Month not selected", 400

    try:
        df = pd.read_csv(f'processed_{selected_month}.csv')
        budgets = load_budgets().get(selected_month, {})
        spending = df.groupby('Category')['Amount'].sum().to_dict()

        insights = []
        for category in budgets:
            spent = spending.get(category, 0)
            budget = budgets[category]

            if spent == 0:
                feedback = f"No expenses recorded yet for {category} this month. Start tracking to get insights!"
                status = "good"
            else:
                status = "overspent" if spent > budget else "good"
                prompt = (
                    f"You are an expert financial coach providing quick feedback on individual budget categories.\n\n"
                    f"The user had set a budget of ${budget:.2f} for the category '{category}' this month, and actually spent ${spent:.2f}.\n\n"
                    "Write a short (2-3 sentences) comment:\n"
                    "- If spending exceeded budget, point it out gently and suggest 1 simple improvement.\n"
                    "- If they stayed within budget, congratulate them and encourage them to continue.\n"
                    "- Avoid just repeating the amounts.\n\n"
                    "Be concise, positive, and motivating."
                )
                feedback = query_llm(prompt)

            insights.append({
                "category": category,
                "spent": spent,
                "feedback": feedback,
                "status": status
            })

        available_months = sorted(load_budgets().keys()) 

        return render_template('insights.html',
            insights=insights,
            budgets=budgets,
            spending=spending,
            available_months=available_months,
            selected_month=selected_month
        )

    except Exception as e:
        return f"Insights page error: {str(e)}"


@app.route('/chart')
def spending_chart():
    selected_month = session.get('selected_month')
    if not selected_month:
        return "Month not selected", 400

    spending = calculate_spending(selected_month)
    return render_template('chart.html', spending=spending)

@app.route('/download_transactions')
def download_transactions():
    selected_month = session.get('selected_month')
    if not selected_month:
        return "Month not selected", 400

    filepath = f'processed_{selected_month}.csv'
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return "File not found", 404

@app.route('/suggest_budget', methods=['POST'])
def suggest_budget():
    try:
        selected_month = session.get('selected_month')
        if not selected_month:
            return jsonify({"error": "No month selected."}), 400

        spending = calculate_spending(selected_month)
        budgets = load_budgets().get(selected_month, {})

        prompt = (
            f"The user has submitted their actual spending vs set budget for the month of {selected_month}.\n\n"
            f"Category-wise Budgets:\n{json.dumps(budgets, indent=2)}\n\n"
            f"Actual Spending:\n{json.dumps(spending, indent=2)}\n\n"
            f"As a financial advisor, analyze where the user overspent or underspent. Then revise the budget "
            f"recommendations accordingly in a JSON format like this:\n"
            f"{{'Food': 350, 'Transportation': 200, ...}}\n\n"
            f"Briefly consider trends or issues. Do not add explanations in the JSON, only the budget values."
        )

        response = query_llm(prompt)

        try:
            suggestion = json.loads(response)
        except json.JSONDecodeError:
            suggestion = {
                cat: round(spending.get(cat, 0) * 1.1, 2)
                for cat in ["Food", "Transportation", "Entertainment", "Shopping", "Other"]
            }

        return jsonify(suggestion)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dashboard')
def dashboard():
    selected_month = session.get('selected_month')
    if not selected_month:
        return "Month not selected", 400

    try:
        df = pd.read_csv(f'processed_{selected_month}.csv')
        budgets = load_budgets().get(selected_month, {})
        spending = df.groupby('Category')['Amount'].sum().to_dict()

        total_budget = sum(budgets.values())
        total_spent = sum(spending.values())
        percent_used = (total_spent / total_budget * 100) if total_budget else 0

        category_usage = {
            cat: {
                "spent": spending.get(cat, 0),
                "budget": budgets[cat],
                "percent": round((spending.get(cat, 0) / budgets[cat]) * 100, 1) if budgets[cat] else 0
            } for cat in budgets
        }

        overall_prompt = (
            f"You are a friendly personal finance coach helping users improve their budgeting habits.\n\n"
            f"The user has set a total monthly budget of ${total_budget:.2f} across categories like Food, Transportation, Entertainment, Shopping, and Other.\n\n"
            f"They actually spent ${total_spent:.2f} this month.\n\n"
            f"Their overall budget usage is {percent_used:.1f}%.\n\n"
            "Write a warm, motivational paragraph (4-5 sentences) summarizing:\n"
            "- Whether they stayed within budget or overspent\n"
            "- Praise if they did well, but give gentle advice if overspending happened\n"
            "- Offer 1 concrete tip for next month (e.g., meal planning, tracking entertainment spend)\n\n"
            "Use an empathetic, encouraging tone. Do not repeat the numbers unnecessarily.\n\n"
        )

        for cat, data in category_usage.items():
            overall_prompt += f"- {cat}: Budget ${data['budget']:.2f}, Spent ${data['spent']:.2f} ({data['percent']}% used)\n"

        summary = query_llm(overall_prompt)

        available_months = sorted(load_budgets().keys())  

        return render_template('dashboard.html',
            total_budget=total_budget,
            total_spent=total_spent,
            percent_used=round(percent_used, 1),
            category_usage=category_usage,
            summary=summary,
            budgets=budgets,
            spending=spending,
            transactions=df.to_dict(orient='records'),
            available_months=available_months,   
            selected_month=selected_month        
        )

    except Exception as e:
        return f"Dashboard error: {str(e)}"


@app.route('/change_month', methods=['POST'])
def change_month():
    selected_month = request.form.get('month')
    if selected_month:
        session['selected_month'] = selected_month
        flash(f"Switched to {selected_month}", "success")
    else:
        flash("No month selected.", "error")

    referer = request.referrer or url_for('dashboard')
    return redirect(referer)

# ---------- Main ----------
if __name__ == '__main__':
    app.run(debug=True)

