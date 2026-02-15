"""
Multi-Agent AI System for Realistic Study Plan Generation

This system uses three specialized agents:
1. Curriculum Agent - Proposes learning roadmap
2. Time Estimation Agent - Validates schedule feasibility  
3. Critic Agent - Flags unrealistic plans

The agents collaborate to produce validated final study plans.
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
from abc import ABC, abstractmethod


@dataclass
class StudyTopic:
    """Represents a single study topic or module"""
    name: str
    description: str
    estimated_hours: float
    prerequisites: List[str]
    difficulty_level: str  # beginner, intermediate, advanced


@dataclass
class StudyPlan:
    """Complete study plan with timeline"""
    title: str
    description: str
    total_duration_weeks: int
    topics: List[StudyTopic]
    weekly_schedule: Dict[int, List[str]]
    daily_hours: float
    validation_score: float
    critic_feedback: List[str]


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        pass


class OllamaClient(LLMClient):
    """Client for Ollama local LLM server"""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
    
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error generating response: {str(e)}"


class OpenRouterClient(LLMClient):
    """Client for OpenRouter cloud API - many free models"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.1-8b-instruct"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
    
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/multi-agent-study-planner",
                "X-Title": "Multi-Agent Study Planner"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7
                },
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating response: {str(e)}"


class GroqClient(LLMClient):
    """Client for Groq cloud API"""
    
    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
    
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7
                },
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


class Agent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, llm_client: LLMClient):
        self.name = name
        self.llm_client = llm_client
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass


class CurriculumAgent(Agent):
    """Generates learning roadmaps and study content"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__("Curriculum Agent", llm_client)
    
    def process(self, learning_goal: str, current_level: str, time_constraint: str) -> Dict[str, Any]:
        """Generate initial curriculum roadmap"""
        
        system_prompt = """You are an expert curriculum designer. Create structured learning paths that are:
        1. Logically sequenced with proper prerequisites
        2. Appropriately challenging for the specified level
        3. Comprehensive yet achievable
        4. Focused on practical application
        
        Always respond with valid JSON format."""
        
        prompt = f"""
        Create a detailed study plan for someone who wants to: {learning_goal}
        
        Current level: {current_level}
        Time available: {time_constraint}
        
        Please provide:
        1. A structured list of topics/modules (5-10 items)
        2. For each topic: name, description, estimated hours, prerequisites, difficulty level
        3. Logical sequencing from fundamentals to advanced
        4. Focus on practical skills and real-world application
        
        Format your response as JSON:
        {{
            "title": "Study Plan Title",
            "description": "Overall description",
            "topics": [
                {{
                    "name": "Topic Name",
                    "description": "Detailed description",
                    "estimated_hours": 20.5,
                    "prerequisites": ["Prerequisite1", "Prerequisite2"],
                    "difficulty_level": "beginner"
                }}
            ]
        }}
        """
        
        response = self.llm_client.generate_response(prompt, system_prompt)
        
        # Extract JSON from response (handle markdown code blocks)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_content = response[json_start:json_end]
        else:
            json_content = response
        
        try:
            curriculum_data = json.loads(json_content)
            return {
                "agent": self.name,
                "status": "success",
                "curriculum": curriculum_data,
                "raw_response": response
            }
        except json.JSONDecodeError:
            return {
                "agent": self.name,
                "status": "error",
                "message": "Failed to parse curriculum response",
                "raw_response": response
            }


class TimeEstimationAgent(Agent):
    """Validates schedule feasibility and time estimates"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__("Time Estimation Agent", llm_client)
    
    def process(self, curriculum: Dict[str, Any], available_hours_per_week: float, deadline_weeks: int) -> Dict[str, Any]:
        """Validate and adjust time estimates"""
        
        total_estimated_hours = sum(topic.get("estimated_hours", 0) for topic in curriculum.get("topics", []))
        total_available_hours = available_hours_per_week * deadline_weeks
        
        system_prompt = """You are a time management expert. Analyze study plans for:
        1. Realistic time estimates
        2. Proper pacing and workload distribution
        3. Achievement within deadlines
        4. Sustainable study schedules
        
        Always respond with valid JSON format."""
        
        prompt = f"""
        Analyze this study plan for time feasibility:
        
        Curriculum: {json.dumps(curriculum, indent=2)}
        
        Constraints:
        - Available hours per week: {available_hours_per_week}
        - Deadline: {deadline_weeks} weeks
        - Total estimated hours: {total_estimated_hours:.1f}
        - Total available hours: {total_available_hours:.1f}
        
        Provide:
        1. Validation score (0-100)
        2. Weekly schedule allocation
        3. Adjusted time estimates if needed
        4. Feasibility assessment
        
        Format as JSON:
        {{
            "validation_score": 85,
            "is_feasible": true,
            "weekly_schedule": {{
                "week_1": ["Topic1", "Topic2"],
                "week_2": ["Topic3"]
            }},
            "adjusted_estimates": [
                {{
                    "topic_name": "Topic Name",
                    "original_hours": 20.5,
                    "adjusted_hours": 18.0,
                    "reason": "Optimized for better pacing"
                }}
            ],
            "feedback": ["Positive feedback", "Suggestions for improvement"]
        }}
        """
        
        response = self.llm_client.generate_response(prompt, system_prompt)
        
        # Extract JSON from response (handle markdown code blocks)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_content = response[json_start:json_end]
        else:
            json_content = response
        
        try:
            time_analysis = json.loads(json_content)
            return {
                "agent": self.name,
                "status": "success",
                "time_analysis": time_analysis,
                "total_hours_comparison": {
                    "estimated": total_estimated_hours,
                    "available": total_available_hours,
                    "difference": total_available_hours - total_estimated_hours
                },
                "raw_response": response
            }
        except json.JSONDecodeError:
            return {
                "agent": self.name,
                "status": "error",
                "message": "Failed to parse time analysis response",
                "raw_response": response
            }


class CriticAgent(Agent):
    """Identifies unrealistic or problematic aspects of study plans"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__("Critic Agent", llm_client)
    
    def process(self, curriculum: Dict[str, Any], time_analysis: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Critique and flag unrealistic aspects"""
        
        system_prompt = """You are a critical learning analyst. Identify potential issues in study plans:
        1. Unrealistic expectations or timelines
        2. Missing prerequisites or knowledge gaps
        3. Overly ambitious workload
        4. Practical feasibility concerns
        5. Resource requirements
        
        Be constructive but thorough. Always respond with valid JSON format."""
        
        prompt = f"""
        Critically analyze this study plan for realism and feasibility:
        
        Curriculum: {json.dumps(curriculum, indent=2)}
        Time Analysis: {json.dumps(time_analysis, indent=2)}
        User Profile: {json.dumps(user_profile, indent=2)}
        
        Identify:
        1. Critical issues (deal-breakers)
        2. Moderate concerns (should be addressed)
        3. Minor suggestions (nice to have)
        4. Overall feasibility rating
        5. Specific recommendations for improvement
        
        Format as JSON:
        {{
            "overall_rating": 75,
            "critical_issues": [
                {{
                    "issue": "Description of critical issue",
                    "impact": "high/medium/low",
                    "suggestion": "How to fix it"
                }}
            ],
            "moderate_concerns": [
                {{
                    "concern": "Description of concern",
                    "suggestion": "How to address it"
                }}
            ],
            "minor_suggestions": [
                "Suggestion 1",
                "Suggestion 2"
            ],
            "final_verdict": "feasible_with_modifications/needs_major_revision/not_feasible",
            "recommendations": ["Actionable recommendation 1", "Actionable recommendation 2"]
        }}
        """
        
        response = self.llm_client.generate_response(prompt, system_prompt)
        
        # Extract JSON from response (handle markdown code blocks)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_content = response[json_start:json_end]
        else:
            json_content = response
        
        try:
            critique = json.loads(json_content)
            return {
                "agent": self.name,
                "status": "success",
                "critique": critique,
                "raw_response": response
            }
        except json.JSONDecodeError:
            return {
                "agent": self.name,
                "status": "error",
                "message": "Failed to parse critique response",
                "raw_response": response
            }


class MultiAgentStudyPlanner:
    """Orchestrates collaboration between all agents"""
    
    def __init__(self, llm_client: LLMClient):
        self.curriculum_agent = CurriculumAgent(llm_client)
        self.time_agent = TimeEstimationAgent(llm_client)
        self.critic_agent = CriticAgent(llm_client)
        self.llm_client = llm_client
    
    def generate_study_plan(self, 
                          learning_goal: str,
                          current_level: str,
                          time_constraint: str,
                          available_hours_per_week: float,
                          deadline_weeks: int,
                          user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate complete validated study plan through agent collaboration"""
        
        if user_profile is None:
            user_profile = {
                "learning_style": "mixed",
                "experience": "beginner",
                "motivation": "high",
                "resources": ["internet", "books"]
            }
        
        print("🚀 Starting Multi-Agent Study Plan Generation...")
        print(f"📚 Goal: {learning_goal}")
        print(f"⏰ Timeline: {deadline_weeks} weeks, {available_hours_per_week}h/week")
        
        # Step 1: Curriculum Agent generates initial roadmap
        print("\n🎓 Step 1: Curriculum Agent generating learning roadmap...")
        curriculum_result = self.curriculum_agent.process(learning_goal, current_level, time_constraint)
        
        if curriculum_result["status"] != "success":
            return {"error": "Curriculum generation failed", "details": curriculum_result}
        
        # Step 2: Time Estimation Agent validates schedule
        print("⏱️ Step 2: Time Estimation Agent validating schedule feasibility...")
        time_result = self.time_agent.process(
            curriculum_result["curriculum"], 
            available_hours_per_week, 
            deadline_weeks
        )
        
        if time_result["status"] != "success":
            return {"error": "Time validation failed", "details": time_result}
        
        # Step 3: Critic Agent reviews and flags issues
        print("🔍 Step 3: Critic Agent analyzing plan for realism...")
        critic_result = self.critic_agent.process(
            curriculum_result["curriculum"],
            time_result["time_analysis"],
            user_profile
        )
        
        if critic_result["status"] != "success":
            return {"error": "Critique failed", "details": critic_result}
        
        # Step 4: Generate final validated plan
        print("✅ Step 4: Generating final validated study plan...")
        final_plan = self._create_final_plan(
            curriculum_result,
            time_result,
            critic_result,
            learning_goal,
            available_hours_per_week,
            deadline_weeks
        )
        
        return {
            "status": "success",
            "final_plan": final_plan,
            "agent_results": {
                "curriculum": curriculum_result,
                "time_analysis": time_result,
                "critique": critic_result
            },
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_processing_time": "multi_agent_collaboration"
            }
        }
    
    def _create_final_plan(self, 
                          curriculum_result: Dict[str, Any],
                          time_result: Dict[str, Any], 
                          critic_result: Dict[str, Any],
                          learning_goal: str,
                          available_hours_per_week: float,
                          deadline_weeks: int) -> StudyPlan:
        """Create the final validated study plan"""
        
        curriculum = curriculum_result["curriculum"]
        time_analysis = time_result["time_analysis"]
        critique = critic_result["critique"]
        
        # Create StudyTopic objects
        topics = []
        for topic_data in curriculum.get("topics", []):
            # Find adjusted estimate if available
            adjusted_hours = topic_data.get("estimated_hours", 0)
            for adj in time_analysis.get("adjusted_estimates", []):
                if adj.get("topic_name") == topic_data.get("name"):
                    adjusted_hours = adj.get("adjusted_hours", adjusted_hours)
                    break
            
            topics.append(StudyTopic(
                name=topic_data.get("name", ""),
                description=topic_data.get("description", ""),
                estimated_hours=adjusted_hours,
                prerequisites=topic_data.get("prerequisites", []),
                difficulty_level=topic_data.get("difficulty_level", "beginner")
            ))
        
        # Extract weekly schedule
        weekly_schedule = time_analysis.get("weekly_schedule", {})
        
        # Create final plan
        return StudyPlan(
            title=curriculum.get("title", f"Study Plan: {learning_goal}"),
            description=curriculum.get("description", ""),
            total_duration_weeks=deadline_weeks,
            topics=topics,
            weekly_schedule=weekly_schedule,
            daily_hours=available_hours_per_week / 7,
            validation_score=time_analysis.get("validation_score", 0),
            critic_feedback=critique.get("recommendations", [])
        )


def main():
    """Example usage of the multi-agent study planner"""
    
    print("🤖 Multi-Agent Study Plan Generator")
    print("=" * 50)
    
    # Initialize LLM client (choose one)
    # Option 1: Local Ollama
    # llm_client = OllamaClient(model_name="llama3.1:8b")
    
    # Option 2: Cloud Groq (uncomment and add API key)
    llm_client = GroqClient(api_key="your-groq-api-key-here")
    
    # Initialize the multi-agent planner
    planner = MultiAgentStudyPlanner(llm_client)
    
    # Example study plan request
    learning_goal = "Learn Python for Data Science and Machine Learning"
    current_level = "Beginner with some programming experience"
    time_constraint = "3 months full-time study"
    available_hours_per_week = 25
    deadline_weeks = 12
    
    user_profile = {
        "learning_style": "visual and hands-on",
        "experience": "some programming in JavaScript",
        "motivation": "high - career change",
        "resources": ["laptop", "internet", "online courses"],
        "preferred_pace": "intensive but sustainable"
    }
    
    # Generate the study plan
    result = planner.generate_study_plan(
        learning_goal=learning_goal,
        current_level=current_level,
        time_constraint=time_constraint,
        available_hours_per_week=available_hours_per_week,
        deadline_weeks=deadline_weeks,
        user_profile=user_profile
    )
    
    if result["status"] == "success":
        print("\n🎉 Study Plan Generated Successfully!")
        print("\n📋 FINAL VALIDATED STUDY PLAN:")
        print("=" * 40)
        
        plan = result["final_plan"]
        print(f"📚 Title: {plan.title}")
        print(f"📝 Description: {plan.description}")
        print(f"⏰ Duration: {plan.total_duration_weeks} weeks")
        print(f"📊 Daily Study Time: {plan.daily_hours:.1f} hours")
        print(f"✅ Validation Score: {plan.validation_score}/100")
        
        print("\n🎯 Study Topics:")
        for i, topic in enumerate(plan.topics, 1):
            print(f"{i}. {topic.name} ({topic.difficulty_level})")
            print(f"   ⏱️  {topic.estimated_hours} hours")
            print(f"   📖 {topic.description}")
            if topic.prerequisites:
                print(f"   🔗 Prerequisites: {', '.join(topic.prerequisites)}")
            print()
        
        print("📅 Weekly Schedule:")
        for week, topics in plan.weekly_schedule.items():
            print(f"{week}: {', '.join(topics)}")
        
        print("\n💡 Critic Recommendations:")
        for feedback in plan.critic_feedback:
            print(f"• {feedback}")
        
        print(f"\n🤖 Generated at: {result['metadata']['generated_at']}")
        
    else:
        print("❌ Error generating study plan:")
        print(result.get("error", "Unknown error"))


if __name__ == "__main__":
    main()
