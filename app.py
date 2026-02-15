"""
Simple Web App Demo for Multi-Agent AI Study Planner
"""

from flask import Flask, render_template, request, jsonify
from multi_agent_study_planner import MultiAgentStudyPlanner, OpenRouterClient
import json
import os

app = Flask(__name__)

# Initialize the planner with OpenRouter
api_key = "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"
llm_client = OpenRouterClient(api_key=api_key)
planner = MultiAgentStudyPlanner(llm_client)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    try:
        # Get form data
        learning_goal = request.form.get('learning_goal')
        current_level = request.form.get('current_level')
        time_constraint = request.form.get('time_constraint')
        available_hours = float(request.form.get('available_hours'))
        deadline_weeks = int(request.form.get('deadline_weeks'))
        
        # Create user profile
        user_profile = {
            "learning_style": request.form.get('learning_style', 'mixed'),
            "experience": current_level.lower(),
            "motivation": request.form.get('motivation', 'medium'),
            "resources": ["laptop", "internet"],
            "preferred_pace": request.form.get('preferred_pace', 'steady')
        }
        
        # Generate study plan
        result = planner.generate_study_plan(
            learning_goal=learning_goal,
            current_level=current_level,
            time_constraint=time_constraint,
            available_hours_per_week=available_hours,
            deadline_weeks=deadline_weeks,
            user_profile=user_profile
        )
        
        if result["status"] == "success":
            plan = result["final_plan"]
            
            # Prepare response data
            response_data = {
                "success": True,
                "plan": {
                    "title": plan.title,
                    "description": plan.description,
                    "duration_weeks": plan.total_duration_weeks,
                    "daily_hours": round(plan.daily_hours, 1),
                    "validation_score": plan.validation_score,
                    "topics": [
                        {
                            "name": topic.name,
                            "description": topic.description,
                            "hours": topic.estimated_hours,
                            "level": topic.difficulty_level,
                            "prerequisites": topic.prerequisites
                        }
                        for topic in plan.topics
                    ],
                    "weekly_schedule": plan.weekly_schedule,
                    "critic_feedback": plan.critic_feedback
                },
                "agent_results": {
                    "curriculum": result["agent_results"]["curriculum"]["status"],
                    "time_analysis": result["agent_results"]["time_analysis"]["status"],
                    "critique": result["agent_results"]["critique"]["status"]
                }
            }
        else:
            response_data = {
                "success": False,
                "error": result.get("error", "Unknown error occurred")
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "Multi-Agent Study Planner"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
