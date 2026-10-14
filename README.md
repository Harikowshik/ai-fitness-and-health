# 🏋️‍♂️ AI Health & Fitness Planner Pro

An intelligent, full-stack health, nutrition, and exercise planning system engineered using **LangGraph Multi-Agent Architecture** and **Google Gemini LLMs**.

This application transforms personal biometrics, dietary restrictions, and fitness goals into deeply personalized daily nutrition blueprints, targeted workout regimens, aisle-categorized grocery checklists, and an interactive AI fitness coach.

---

## 📑 Table of Contents

- [Architectural Overview](#-architectural-overview)
- [System Workflow Diagram](#-system-workflow-diagram)
- [Deep Dive: Core Concepts & Methodologies](#-deep-dive-core-concepts--methodologies)
  - [1. LangGraph Multi-Agent Architecture (DAG StateGraph)](#1-langgraph-multi-agent-architecture-dag-stategraph)
  - [2. Biometric & Nutritional Science Engine](#2-biometric--nutritional-science-engine)
  - [3. Multi-Model Fallback & API Resilience](#3-multi-model-fallback--api-resilience)
  - [4. Conversational Coaching & Contextual Q&A](#4-conversational-coaching--contextual-qa)
  - [5. Smart Grocery List Extraction](#5-smart-grocery-list-extraction)
- [Key Features Breakdown](#-key-features-breakdown)
- [File Structure](#-file-structure)
- [Step-by-Step Installation & Setup](#-step-by-step-installation--setup)
- [User Guide: Running the Application](#-user-guide-running-the-application)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🏛️ Architectural Overview

Rather than relying on a single monolithic prompt to handle nutrition, workouts, and groceries, this system uses a **multi-agent orchestration pattern**. Specialization ensures each domain receives precise, expert-level system prompts, producing higher quality, clinically sensible recommendations.

```
                                 ┌─────────────────────────┐
                                 │   User Profile Input    │
                                 │ (Biometrics, Goal, Diet)│
                                 └───────────┬─────────────┘
                                             │
                         ┌───────────────────▼───────────────────┐
                         │   Biometric & Macro Calculation Engine │
                         │     (BMI, BMR, TDEE, Macros, Water)   │
                         └───────────────────┬───────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │ (Parallel Fan-Out)                            │
          ┌──────────▼──────────┐                         ┌──────────▼──────────┐
          │  Dietary Specialist │                         │ Fitness Coach Agent │
          │        Agent        │                         │                     │
          │ (Meals & Nutrition) │                         │ (Routines & Splits) │
          └──────────┬──────────┘                         └──────────┬──────────┘
                     │ (Generated Meal Plan)                         │
          ┌──────────▼──────────┐                                    │
          │  Grocery Specialist │                                    │
          │        Agent        │                                    │
          │ (Aisle Categorized) │                                    │
          └──────────┬──────────┘                                    │
                     │ (Parallel Fan-In)                             │
                     └───────────────────────┬───────────────────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │ Streamlit Dashboard │
                                  │   (Tabs, Metrics,   │
                                  │  Timer, Export, Q&A)│
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │ Interactive AI Coach│
                                  │ (Contextual Q&A)    │
                                  └─────────────────────┘
```

---

## 🔬 Deep Dive: Core Concepts & Methodologies

### 1. LangGraph Multi-Agent Architecture (DAG StateGraph)

**What is LangGraph?**
[LangGraph](https://github.com/langchain-ai/langgraph) is an enterprise-standard framework designed by LangChain for coordinating multi-actor agent workflows using directed graph structures (nodes, edges, and state).

**Why a StateGraph DAG instead of a single prompt or linear script?**
- **Stateful Typed State (`TypedDict`)**: All agents write to and read from a typed `HealthPlanState`, guaranteeing strict contracts between agents.
- **Concurrent Fan-Out / Fan-In Execution**: `START` branches out concurrently into both `dietary_specialist` and `fitness_coach`. The `grocery_specialist` node runs as a downstream dependency of the dietary node. This parallelism reduces latency while preserving data flow.
- **Domain Specialization**: The Dietary Specialist focuses on nutritional chemistry and macro targets; the Fitness Coach focuses on CSCS biomechanics; the Grocery Specialist parses ingredients into aisle categories.

---

### 2. Biometric & Nutritional Science Engine

The application does not ask the LLM to guess basic arithmetic. It calculates baseline metrics using validated physiological formulas before feeding the values to the agents:

#### A. Body Mass Index (BMI)
$$\text{BMI} = \frac{\text{Weight (kg)}}{(\text{Height (m)})^2}$$
Classifies the user into *Underweight*, *Normal Weight*, *Overweight*, or *Obese*.

#### B. Basal Metabolic Rate (BMR) – The Mifflin-St Jeor Equation
Considered the gold standard in clinical settings for estimating resting caloric burn:
- **For Males**:
  $$\text{BMR} = (10 \times \text{weight in kg}) + (6.25 \times \text{height in cm}) - (5 \times \text{age}) + 5$$
- **For Females**:
  $$\text{BMR} = (10 \times \text{weight in kg}) + (6.25 \times \text{height in cm}) - (5 \times \text{age}) - 161$$

#### C. Total Daily Energy Expenditure (TDEE)
$$\text{TDEE} = \text{BMR} \times \text{Activity Multiplier}$$
- **Sedentary** (desk job, minimal movement): $\times 1.2$
- **Lightly Active** (1–3 days exercise/week): $\times 1.375$
- **Moderately Active** (3–5 days exercise/week): $\times 1.55$
- **Very Active** (6–7 days hard exercise): $\times 1.725$
- **Extremely Active** (endurance training or physical job): $\times 1.9$

#### D. Caloric Adjustment by Fitness Goal
- **Lose Weight**: Target is set to $\text{TDEE} - 500\text{ kcal}$ (a standard deficit for ~0.5 kg fat loss/week).
- **Gain Muscle**: Target is set to $\text{TDEE} + 350\text{ kcal}$ (lean caloric surplus).
- **Strength Training**: Target is set to $\text{TDEE} + 250\text{ kcal}$.
- **Stay Fit / Maintain**: Target equals $\text{TDEE}$.

#### E. Macronutrient Distribution Formulas
- **Protein**: Prioritized at $1.8\text{ to }2.2\text{ g per kg}$ of bodyweight to preserve lean muscle during deficits and stimulate muscle protein synthesis (MPS) during surplus.
- **Healthy Fats**: Set at $0.8\text{ to }1.0\text{ g per kg}$ to support endocrine function and hormone production.
- **Carbohydrates**: Remaining daily calories are allocated to carbs (4 kcal/g) to fuel glycogen stores and training intensity.
- **Keto / Low-Carb Adjustment**: Automatically shifts to 68% fats, 24% protein, and 8% carbohydrates.

#### F. Daily Hydration Formula
$$\text{Daily Water (Liters)} = (\text{Weight in kg} \times 0.033) + \text{Activity Offset (0.5L)}$$

---

### 3. Multi-Model Fallback & API Resilience

LLM APIs can experience temporary spikes in demand, returning `503 UNAVAILABLE` or quota limit responses. 

To prevent errors from disrupting user experience, `health_agent.py` implements **`run_agent_with_fallback`**:
1. It attempts execution with the primary model (**`gemini-3.6-flash`**).
2. It inspects both exceptions and raw string outputs for error signatures (`{"error": ...}`, `code 503`, `UNAVAILABLE`, `code 429`).
3. If an error is detected, it automatically cascades through a backup pool:
   $$\text{gemini-3.6-flash} \longrightarrow \text{gemini-3.8-flash} \longrightarrow \text{gemini-3.7-flash} \longrightarrow \text{gemini-2.5-pro}$$
4. The user receives their generated plan uninterrupted, with an informative badge noting which model handled the request.

---

### 4. Conversational Coaching & Contextual Q&A

The **Interactive Coach Agent** does not operate in isolation. When the user asks a question, the application automatically injects the **current active meal plan** and **current workout routine** into the agent's context window.

This allows the coach to answer contextual questions accurately:
- *"What can I substitute for the protein in Tuesday's lunch?"*
- *"Can I swap the barbell deadlifts for dumbbells if I have lower back pain?"*
- *"How should I adjust my calories on rest days?"*

---

### 5. Smart Grocery List Extraction

A dedicated **Grocery Specialist Agent** parses the unstructured markdown meal plan and extracts raw ingredients. It formats them into a shopping checklist grouped into supermarket aisles:
- 🥦 **Fresh Produce**: Vegetables, fruits, herbs.
- 🥩 **Proteins & Dairy / Plant Alternatives**: Chicken, tofu, Greek yogurt, eggs, whey.
- 🌾 **Pantry & Grains**: Oats, quinoa, brown rice, whole wheat wraps.
- 🫒 **Healthy Fats & Condiments**: Olive oil, peanut butter, chia seeds, spices.

---

## 🌟 Key Features Breakdown

| Feature | Description |
| :--- | :--- |
| **Tri-Agent System** | Separate Dietary, Fitness, and Grocery agents orchestrated by Agno. |
| **1-Click Presets** | Instant profile loaders for *Fat Loss*, *Muscle Hypertrophy*, *Keto*, and *Plant-Based*. |
| **Nutritional Dashboard** | Real-time calculations of BMI, TDEE, Calorie targets, and gram-level macro splits. |
| **Categorized Grocery List** | Checklist-ready grocery list extracted directly from the meal blueprint. |
| **Hydration Tracker** | Interactive water log with a visual progress bar and `+1 Glass (250ml)` button. |
| **Rest Interval Timer** | On-screen interval countdown (30s, 60s, 90s, 120s) for pacing workout sets. |
| **1-Click Quick Modifiers** | Instant prompts for *15-Min Prep*, *Home Gym*, *Budget Swaps*, and *Vegan Swaps*. |
| **Markdown Export** | Single-click download of the complete blueprint and grocery checklist. |

---

## 📁 File Structure

```
AI Fitness And Health/
├── .env                # Stores GEMINI_API_KEY securely
├── health_agent.py     # Main application: UI, Agents, Logic & Dashboard
├── requirements.txt    # Production dependencies
└── README.md           # Detailed architecture & execution documentation
```

---

## 🛠️ Step-by-Step Installation & Setup

### 1. Prerequisites
- **Python 3.10 to 3.13** installed on your system.
- A free **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/apikey).

### 2. Install Dependencies
Open your terminal in this directory and run:

```bash
pip install -r requirements.txt
```

### 3. Configure Your Gemini API Key
Add your API key to the `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Note: You can also enter or update your key anytime from the application's sidebar).*

---

## 🚀 User Guide: Running the Application

### Launch the Streamlit App
Run the following command:

```bash
streamlit run health_agent.py
```

Streamlit will open the app in your browser at:
**`http://localhost:8501`**

### Recommended Workflow:
1. **Load a Preset or Enter Biometrics**: Use the sidebar presets or enter your age, weight, height, activity level, and goals.
2. **Review Metrics**: Check your baseline BMI, maintenance calories, target calories, and macro breakdown.
3. **Generate Blueprint**: Click **`🚀 Generate My Complete Health & Fitness Blueprint`**.
4. **Explore the Tabs**:
   - 🥗 **Meal Blueprint**: Review your meals, portions, and considerations.
   - 🏋️‍♂️ **Workout Regimen**: Follow your workout split, sets, reps, and form tips.
   - 🛒 **Smart Grocery List**: Check off items needed for prep.
   - 💧 **Hydration Tracker**: Log your daily water intake.
   - ⏱️ **Rest Timer**: Time your rest periods during your workout.
5. **Ask Follow-Up Questions**: Use the 1-click modifier buttons or type custom questions to your AI Coach.
6. **Download Plan**: Export your plan as a Markdown file for offline reference.

---

## ❓ Troubleshooting & FAQ

#### 1. What if I get a `503 UNAVAILABLE` error?
Google's preview models occasionally experience traffic surges. The application has **automatic fallback resilience** built-in. If one model is busy, it automatically routes to alternative Gemini models without failing.

#### 2. Can I use this at home without gym equipment?
Yes! Type *"home dumbbells only"* in the **Allergies/Equipment/Injuries** input, or click the **`🏠 Home Gym (Dumbbells Only)`** button under Quick Plan Modifications.

#### 3. How do I change my dietary preference to Vegan or Keto?
Select your framework from the **Dietary Framework** dropdown. The biometrics engine will recalibrate your macronutrient split and meal blueprints automatically.
