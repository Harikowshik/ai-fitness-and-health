import os
import json
import time
import warnings
import streamlit as st
from dotenv import load_dotenv
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.models.google import Gemini

warnings.filterwarnings("ignore")
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Health & Fitness Planner Pro",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern, premium appearance
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.02rem;
        color: #cbd5e1;
        line-height: 1.55;
        max-width: 900px;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
    }

    .stButton>button {
        font-weight: 700;
        letter-spacing: 0.02em;
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 6px;
        margin-right: 0.3rem;
    }
    .badge-blue { background: #0284c7; color: white; }
    .badge-green { background: #16a34a; color: white; }
    .badge-amber { background: #d97706; color: white; }
    </style>
""", unsafe_allow_html=True)

def calculate_health_metrics(age, weight, height, sex, activity_level, goal, diet):
    """Calculates BMI, BMR, TDEE, target calories, macro distribution and daily water goal."""
    height_m = height / 100.0
    bmi = round(weight / (height_m ** 2), 1) if height_m > 0 else 0
    
    # Mifflin-St Jeor Equation
    if sex == "Male":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
        
    activity_factors = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Extremely Active": 1.9
    }
    multiplier = activity_factors.get(activity_level, 1.375)
    tdee = round(bmr * multiplier)
    
    # Target Calories
    if goal == "Lose Weight":
        target_calories = max(1200, tdee - 500)
    elif goal == "Gain Muscle":
        target_calories = tdee + 350
    elif goal == "Strength Training":
        target_calories = tdee + 250
    else:
        target_calories = tdee
        
    # BMI Category
    if bmi < 18.5:
        bmi_status = "Underweight"
    elif bmi < 25:
        bmi_status = "Normal Weight"
    elif bmi < 30:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"
        
    # Macronutrient Breakdown
    if "Keto" in diet or "Low Carb" in diet:
        carb_cals = target_calories * 0.08
        fat_cals = target_calories * 0.68
        prot_cals = target_calories * 0.24
    elif goal == "Lose Weight":
        protein_g = min(weight * 2.1, target_calories * 0.35 / 4)
        prot_cals = protein_g * 4
        fat_g = weight * 0.8
        fat_cals = fat_g * 9
        carb_cals = max(0, target_calories - prot_cals - fat_cals)
    elif goal == "Gain Muscle":
        protein_g = weight * 2.0
        prot_cals = protein_g * 4
        fat_g = weight * 1.0
        fat_cals = fat_g * 9
        carb_cals = max(0, target_calories - prot_cals - fat_cals)
    else:
        protein_g = weight * 1.8
        prot_cals = protein_g * 4
        fat_g = weight * 0.9
        fat_cals = fat_g * 9
        carb_cals = max(0, target_calories - prot_cals - fat_cals)
        
    protein_grams = round(prot_cals / 4)
    fat_grams = round(fat_cals / 9)
    carb_grams = round(carb_cals / 4)
    
    # Daily Water Target (Liters)
    water_liters = round((weight * 0.033) + (0.5 if multiplier > 1.375 else 0.0), 1)
        
    return {
        "bmi": bmi,
        "bmi_status": bmi_status,
        "tdee": tdee,
        "target_calories": target_calories,
        "protein_g": protein_grams,
        "carb_g": carb_grams,
        "fat_g": fat_grams,
        "water_l": water_liters
    }

def run_agent_with_fallback(agent_factory, prompt, api_key, preferred_model):
    """
    Executes an Agno agent with automatic fallback across models.
    Catches 503 high-demand errors or JSON error strings and retries with alternative Gemini models.
    """
    candidate_models = [preferred_model]
    backup_pool = ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-2.5-pro"]
    for m in backup_pool:
        if m not in candidate_models:
            candidate_models.append(m)
            
    last_err = ""
    for candidate in candidate_models:
        try:
            model = Gemini(id=candidate, api_key=api_key)
            agent = agent_factory(model)
            response: RunOutput = agent.run(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Detect JSON error responses returned as text content (e.g. 503 high demand or 429 quota)
            if isinstance(content, str):
                trimmed = content.strip()
                if (trimmed.startswith('{"error"') or trimmed.startswith("{ 'error'") or 
                    '"code": 503' in trimmed or 'UNAVAILABLE' in trimmed or 
                    '"code": 429' in trimmed or 'RESOURCE_EXHAUSTED' in trimmed):
                    last_err = f"Model {candidate} is busy ({trimmed[:80]}...)"
                    continue
                
            return content, candidate
        except Exception as ex:
            last_err = f"Model {candidate} error: {str(ex)}"
            continue
            
    raise RuntimeError(f"All Gemini models were temporarily busy. Details: {last_err}")

def main():
    # Session state initialization
    if 'dietary_plan' not in st.session_state:
        st.session_state.dietary_plan = {}
        st.session_state.fitness_plan = {}
        st.session_state.grocery_list = ""
        st.session_state.qa_pairs = []
        st.session_state.plans_generated = False
        st.session_state.user_summary = ""
        st.session_state.water_logged_ml = 0

    # Presets for quick-loading profiles
    PRESETS = {
        "🔥 Fat Loss & Lean Toning": {
            "age": 28, "sex": "Female", "height": 165.0, "weight": 68.0,
            "activity": "Lightly Active", "diet": "Standard Balanced", "goal": "Lose Weight", "notes": ""
        },
        "💪 Muscle Hypertrophy": {
            "age": 24, "sex": "Male", "height": 180.0, "weight": 74.0,
            "activity": "Very Active", "diet": "High Protein", "goal": "Gain Muscle", "notes": "Access to full gym"
        },
        "🥑 Low-Carb / Keto Athlete": {
            "age": 34, "sex": "Male", "height": 175.0, "weight": 82.0,
            "activity": "Moderately Active", "diet": "Keto / Low Carb", "goal": "Lose Weight", "notes": "No dairy"
        },
        "🌱 Plant-Powered Fitness": {
            "age": 26, "sex": "Female", "height": 170.0, "weight": 62.0,
            "activity": "Moderately Active", "diet": "Vegetarian", "goal": "Stay Fit & Toned", "notes": "Desk worker"
        }
    }

    # Header / Hero banner
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">🏋️‍♂️ AI Health & Fitness Planner Pro</div>
            <div class="hero-subtitle">
                Powered by <strong>Agno AI Multi-Agent Architecture</strong> &amp; <strong>Google Gemini</strong>.
                Precision meal plans, workout splits, smart grocery list extraction, macro targeting, and conversational coaching.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1517838277536-f5f99be501cd?w=600&auto=format&fit=crop&q=80", use_container_width=True)
        st.header("⚙️ Agent Settings")
        
        env_key = os.getenv("GEMINI_API_KEY", "")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=env_key,
            type="password",
            help="Enter your Gemini API key from Google AI Studio"
        )
        
        # gemini-3.6-flash is primary default because it has stable capacity
        model_name = st.selectbox(
            "Gemini Model",
            options=["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-2.5-pro"],
            index=0,
            help="Select the Gemini model variant. Automatic fallback is active if any model is busy."
        )
        
        if not gemini_api_key:
            st.warning("⚠️ Please provide a Gemini API Key to proceed.")
            st.markdown("[Get your free Gemini API Key](https://aistudio.google.com/apikey)")
            return
        else:
            st.success("🟢 API Key Loaded & Active")
            
        st.divider()
        st.subheader("⚡ Quick-Load Presets")
        st.caption("Click to prefill realistic biometrics:")
        for preset_name, p_data in PRESETS.items():
            if st.button(preset_name, use_container_width=True):
                st.session_state.current_preset = p_data
                st.rerun()

    # Determine default field values
    curr = st.session_state.get("current_preset", {})
    default_age = curr.get("age", 25)
    default_sex = curr.get("sex", "Male")
    default_height = curr.get("height", 175.0)
    default_weight = curr.get("weight", 72.0)
    default_act = curr.get("activity", "Moderately Active")
    default_diet = curr.get("diet", "Vegetarian")
    default_goal = curr.get("goal", "Lose Weight")
    default_notes = curr.get("notes", "")

    # Profile Configuration
    st.subheader("👤 Step 1: Your Profile & Biometrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=12, max_value=100, value=default_age, step=1)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=default_height, step=0.5)
        
        act_options = ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"]
        act_idx = act_options.index(default_act) if default_act in act_options else 2
        activity_level = st.selectbox("Daily Activity Level", options=act_options, index=act_idx)
        
        diet_options = ["Standard Balanced", "High Protein", "Vegetarian", "Vegan", "Keto / Low Carb", "Mediterranean", "Gluten-Free", "Dairy-Free"]
        diet_idx = diet_options.index(default_diet) if default_diet in diet_options else 2
        dietary_preferences = st.selectbox("Dietary Framework", options=diet_options, index=diet_idx)

    with col2:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=default_weight, step=0.5)
        sex_options = ["Male", "Female", "Other"]
        sex_idx = sex_options.index(default_sex) if default_sex in sex_options else 0
        sex = st.selectbox("Sex", options=sex_options, index=sex_idx)
        
        goal_options = ["Lose Weight", "Gain Muscle", "Endurance & Stamina", "Stay Fit & Toned", "Strength Training"]
        goal_idx = goal_options.index(default_goal) if default_goal in goal_options else 0
        fitness_goals = st.selectbox("Primary Fitness Goal", options=goal_options, index=goal_idx)
        
        medical_notes = st.text_input(
            "Allergies, Equipment, or Injuries (Optional)",
            value=default_notes,
            placeholder="e.g. Peanut allergy, home dumbbells only, lower back issue..."
        )

    # Health & Nutrition Metrics Dashboard
    m = calculate_health_metrics(age, weight, height, sex, activity_level, fitness_goals, dietary_preferences)
    
    st.markdown("#### 📊 Target Daily Nutrition & Biometrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("BMI", f"{m['bmi']}", m["bmi_status"])
    with c2:
        st.metric("Maintenance (TDEE)", f"{m['tdee']} kcal")
    with c3:
        st.metric("Target Caloric Intake", f"{m['target_calories']} kcal", f"{m['target_calories'] - m['tdee']:+d} kcal")
    with c4:
        st.metric("Hydration Goal", f"{m['water_l']} L / day")

    # Macro targets breakdown
    st.caption("Target Macronutrient Split:")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.info(f"🥩 **Protein**: {m['protein_g']}g (~{round(m['protein_g']*4*100/m['target_calories'])}%)")
    with mc2:
        st.success(f"🥑 **Healthy Fats**: {m['fat_g']}g (~{round(m['fat_g']*9*100/m['target_calories'])}%)")
    with mc3:
        st.warning(f"🌾 **Carbohydrates**: {m['carb_g']}g (~{round(m['carb_g']*4*100/m['target_calories'])}%)")

    st.write("")
    generate_btn = st.button("🚀 Generate My Complete Health & Fitness Blueprint", type="primary", use_container_width=True)

    if generate_btn:
        with st.spinner("🤖 Coordinating Dietary, Fitness & Grocery AI Agents with Gemini..."):
            try:
                def make_dietary_agent(mdl):
                    return Agent(
                        name="Dietary Expert",
                        role="Provides personalized dietary, macro, and meal recommendations",
                        model=mdl,
                        instructions=[
                            "You are an elite registered sports dietitian.",
                            "Consider user biometrics, macro targets, dietary preference, and goal.",
                            "Structure a realistic, appetizing daily meal plan: Breakfast, Lunch, Dinner, and 1-2 Smart Snacks.",
                            "List key ingredients, approximate portion sizes, and estimated protein/carbs/fat for each meal.",
                            "Strictly adhere to any allergy or restriction stated by the user.",
                            "Provide actionable, clean Markdown output.",
                        ]
                    )

                def make_fitness_agent(mdl):
                    return Agent(
                        name="Fitness Expert",
                        role="Provides personalized fitness routines and progressive overload splits",
                        model=mdl,
                        instructions=[
                            "You are a certified strength and conditioning coach (CSCS).",
                            "Formulate a structured weekly workout schedule tailored to the user's goal and equipment availability.",
                            "Include Warm-up, Main Exercises (sets, reps, rest intervals), and Cool-down.",
                            "Highlight key form safety cues and progressive overload principles.",
                            "Tailor volume and exercises away from any noted injuries.",
                        ]
                    )

                def make_grocery_agent(mdl):
                    return Agent(
                        name="Grocery Specialist",
                        role="Extracts categorized grocery shopping items",
                        model=mdl,
                        instructions=[
                            "Extract all ingredients needed to prepare the meals in the dietary plan.",
                            "Group them neatly under: 🥦 Fresh Produce, 🥩 Proteins & Dairy/Plant Alternatives, 🌾 Pantry & Grains, 🫒 Healthy Fats & Condiments.",
                            "Format as a clean markdown checklist (- [ ] Item (est. qty)).",
                        ]
                    )

                user_profile = f"""
                User Biometrics & Targets:
                - Age: {age} | Sex: {sex} | Height: {height}cm | Weight: {weight}kg
                - Body Mass Index: {m['bmi']} ({m['bmi_status']})
                - Activity Level: {activity_level}
                - Dietary Framework: {dietary_preferences}
                - Primary Goal: {fitness_goals}
                - Maintenance Calories (TDEE): ~{m['tdee']} kcal/day
                - Suggested Target Calories: ~{m['target_calories']} kcal/day
                - Macro Targets: Protein {m['protein_g']}g | Fats {m['fat_g']}g | Carbs {m['carb_g']}g
                - Water Target: ~{m['water_l']} Liters/day
                - Special Notes / Limitations: {medical_notes if medical_notes else "None specified"}
                """

                # Run Dietary & Fitness agents with fallback resilience
                dietary_content, dietary_used = run_agent_with_fallback(
                    make_dietary_agent, user_profile, gemini_api_key, model_name
                )
                fitness_content, fitness_used = run_agent_with_fallback(
                    make_fitness_agent, user_profile, gemini_api_key, model_name
                )
                grocery_content, grocery_used = run_agent_with_fallback(
                    make_grocery_agent, f"Extract grocery list from this meal plan:\n{dietary_content}", gemini_api_key, model_name
                )

                if dietary_used != model_name:
                    st.info(f"ℹ️ Model notice: Switched to **{dietary_used}** for stable execution.")

                st.session_state.dietary_plan = {
                    "why_this_plan_works": f"Precision {m['target_calories']} kcal intake with {m['protein_g']}g protein tailored for {fitness_goals} ({dietary_preferences}).",
                    "meal_plan": dietary_content,
                    "important_considerations": f"""
                    • Hydration Target: Aim for at least {m['water_l']} Liters of water daily
                    • Protein Distribution: Distribute {m['protein_g']}g across 3-4 meals to optimize protein synthesis
                    • Electrolyte Balance: Maintain sodium, potassium, and magnesium levels
                    • Listen to Satiety: Adjust vegetable portions freely for fiber & fullness
                    """
                }

                st.session_state.fitness_plan = {
                    "goals": f"Progressive overload stimulus tailored for {fitness_goals}",
                    "routine": fitness_content,
                    "tips": """
                    • Progressive Overload: Aim to increase reps or weight gradually week over week
                    • Recovery Window: Prioritize 7-9 hours of restful sleep every night
                    • Strict Form: Control the eccentric (lowering) phase of each exercise
                    • Hydration: Drink 500ml water 30-45 minutes before training
                    """
                }

                st.session_state.grocery_list = grocery_content
                st.session_state.user_summary = user_profile
                st.session_state.water_target_l = m['water_l']
                st.session_state.plans_generated = True
                st.session_state.qa_pairs = []

            except Exception as e:
                st.error(f"❌ Error generating plan: {e}")

    # Display generated plans in organized tabs
    if st.session_state.plans_generated:
        st.divider()
        st.subheader("📋 Step 2: Your Custom Health & Fitness Blueprint")

        tab_diet, tab_fit, tab_groc, tab_water, tab_timer = st.tabs([
            "🥗 Meal Blueprint",
            "🏋️‍♂️ Workout Regimen",
            "🛒 Smart Grocery List",
            "💧 Hydration Tracker",
            "⏱️ Rest Interval Timer"
        ])

        with tab_diet:
            st.markdown("### 🎯 Why This Plan Works")
            st.info(st.session_state.dietary_plan.get("why_this_plan_works"))
            st.markdown("### 🍽️ Daily Meal Blueprint")
            st.markdown(st.session_state.dietary_plan.get("meal_plan"))
            st.markdown("### ⚠️ Key Nutrition Rules")
            for c in st.session_state.dietary_plan.get("important_considerations", "").split('\n'):
                if c.strip():
                    st.warning(c.strip())

        with tab_fit:
            st.markdown("### 🎯 Training Objectives")
            st.success(st.session_state.fitness_plan.get("goals"))
            st.markdown("### 🏃‍♂️ Workout Program")
            st.markdown(st.session_state.fitness_plan.get("routine"))
            st.markdown("### 💡 Coach's Pro Tips")
            for t in st.session_state.fitness_plan.get("tips", "").split('\n'):
                if t.strip():
                    st.info(t.strip())

        with tab_groc:
            st.markdown("### 🛒 Smart Grocery Shopping List")
            st.caption("Aisle-categorized ingredients for meal prep:")
            st.markdown(st.session_state.grocery_list)

        with tab_water:
            st.markdown("### 💧 Daily Water & Hydration Tracker")
            target_l = st.session_state.get("water_target_l", 3.0)
            target_ml = target_l * 1000
            current_ml = st.session_state.water_logged_ml
            pct = min(1.0, current_ml / target_ml) if target_ml > 0 else 0
            
            st.progress(pct)
            st.write(f"**Logged**: {current_ml} ml / **Goal**: {int(target_ml)} ml ({round(pct*100)}% complete)")
            
            w_col1, w_col2, w_col3 = st.columns([1, 1, 2])
            with w_col1:
                if st.button("➕ 1 Glass (250ml)"):
                    st.session_state.water_logged_ml += 250
                    st.rerun()
            with w_col2:
                if st.button("🔄 Reset Tracker"):
                    st.session_state.water_logged_ml = 0
                    st.rerun()

        with tab_timer:
            st.markdown("### ⏱️ Workout Rest Interval Timer")
            st.caption("Time your rest between working sets:")
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            timer_duration = None
            if t_col1.button("⏳ 30s Rest"): timer_duration = 30
            if t_col2.button("⏳ 60s Rest"): timer_duration = 60
            if t_col3.button("⏳ 90s Rest"): timer_duration = 90
            if t_col4.button("⏳ 120s Rest"): timer_duration = 120
            
            if timer_duration:
                timer_holder = st.empty()
                progress_bar = st.progress(1.0)
                for sec in range(timer_duration, -1, -1):
                    timer_holder.metric("Rest Remaining", f"{sec} seconds")
                    progress_bar.progress(sec / timer_duration if timer_duration > 0 else 0)
                    time.sleep(1)
                timer_holder.success("🔔 Rest over! Start your next set!")

        # One-Click Quick Tweak Buttons
        st.divider()
        st.subheader("⚡ Quick Plan Modifications")
        st.caption("Click any modifier to ask the AI coach for instant customization:")
        q1, q2, q3, q4 = st.columns(4)
        quick_prompt = None
        if q1.button("⏱️ 15-Min Quick Prep Only"):
            quick_prompt = "How can I adapt this meal plan so every meal takes 15 minutes or less to prepare?"
        if q2.button("🏠 Home Gym (Dumbbells Only)"):
            quick_prompt = "Please modify this workout routine so I can do it entirely at home with just dumbbells and bodyweight."
        if q3.button("💰 Budget-Friendly Swaps"):
            quick_prompt = "What are the most cost-effective and budget-friendly ingredient substitutions for this meal plan?"
        if q4.button("🌿 100% Plant-Based Swaps"):
            quick_prompt = "Can you provide plant-based / vegan protein substitutes for all protein sources in this meal plan?"

        # Combined Markdown Export
        combined_report = f"""# Personalized AI Health & Fitness Blueprint Pro
Generated with Agno AI Agent & Google Gemini

## User Profile & Targets
{st.session_state.user_summary}

---

## 🥗 Dietary Blueprint
### Why This Plan Works
{st.session_state.dietary_plan.get('why_this_plan_works')}

### Meal Plan
{st.session_state.dietary_plan.get('meal_plan')}

### Important Considerations
{st.session_state.dietary_plan.get('important_considerations')}

---

## 🏋️‍♂️ Fitness Regimen
### Objectives
{st.session_state.fitness_plan.get('goals')}

### Workout Routine
{st.session_state.fitness_plan.get('routine')}

### Coaching Tips
{st.session_state.fitness_plan.get('tips')}

---

## 🛒 Smart Grocery Shopping List
{st.session_state.grocery_list}
"""
        st.download_button(
            label="📥 Download Complete Plan + Grocery List (Markdown)",
            data=combined_report,
            file_name="my_personalized_health_and_fitness_plan.md",
            mime="text/markdown",
            use_container_width=True
        )

        # Interactive Q&A Coach
        st.divider()
        st.subheader("💬 Step 3: Ask Your AI Health & Fitness Coach")
        st.caption("Ask questions about substitutions, exercise technique, nutrition timing, or program adjustments.")
        
        with st.form(key="qa_form", clear_on_submit=True):
            user_question = st.text_input("Ask a follow-up question:", placeholder="e.g. What can I substitute for the lunch protein? or How can I do squats with knee pain?")
            submit_question = st.form_submit_button("Ask Coach")

        question_to_run = quick_prompt if quick_prompt else (user_question if submit_question else None)

        if question_to_run:
            with st.spinner("AI Coach is formulating your advice..."):
                try:
                    def make_qa_agent(mdl):
                        return Agent(
                            name="Fitness & Nutrition Coach",
                            model=mdl,
                            instructions=[
                                "You are an elite, supportive certified fitness and nutrition coach.",
                                "Answer using the context of the user's generated dietary and fitness plan.",
                                "Provide clear, concise, actionable, and encouraging recommendations.",
                            ]
                        )

                    context = f"""
                    User Dietary Plan:
                    {st.session_state.dietary_plan.get('meal_plan', '')}

                    User Fitness Plan:
                    {st.session_state.fitness_plan.get('routine', '')}

                    User Question: {question_to_run}
                    """

                    answer, qa_used = run_agent_with_fallback(
                        make_qa_agent, context, gemini_api_key, model_name
                    )
                    st.session_state.qa_pairs.insert(0, (question_to_run, answer))
                except Exception as e:
                    st.error(f"❌ Error getting answer: {e}")

        if st.session_state.qa_pairs:
            st.markdown("### 💬 Coaching Conversation History")
            for q, a in st.session_state.qa_pairs:
                with st.chat_message("user"):
                    st.write(q)
                with st.chat_message("assistant"):
                    st.markdown(a)

if __name__ == "__main__":
    main()
