"""
Demo script showing the multi-agent study planner in action
"""

from multi_agent_study_planner import MultiAgentStudyPlanner, OpenRouterClient, GroqClient
import json

def demo_study_planner():
    """Demonstrate the multi-agent system with different learning scenarios"""
    
    print("🎓 Multi-Agent Study Planner Demo")
    print("=" * 50)
    
    # Initialize with OpenRouter (working setup)
    try:
        api_key = "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"
        llm_client = OpenRouterClient(api_key=api_key)
        print("✅ Connected to OpenRouter API")
    except:
        print("❌ Could not connect to Ollama. Make sure it's running with 'ollama serve'")
        print("💡 Alternatively, uncomment the Groq client below and add your API key")
        return
    
    planner = MultiAgentStudyPlanner(llm_client)
    
    # Demo scenarios
    scenarios = [
        {
            "name": "Python Data Science Path",
            "learning_goal": "Learn Python for Data Science and Machine Learning",
            "current_level": "Beginner with some programming experience",
            "time_constraint": "3 months part-time study",
            "available_hours_per_week": 15,
            "deadline_weeks": 12,
            "user_profile": {
                "learning_style": "visual and hands-on",
                "experience": "some programming in JavaScript",
                "motivation": "high - career change",
                "resources": ["laptop", "internet", "online courses"],
                "preferred_pace": "steady and consistent"
            }
        },
        {
            "name": "Web Development Bootcamp",
            "learning_goal": "Become a Full-Stack Web Developer",
            "current_level": "Complete beginner to programming",
            "time_constraint": "6 months intensive study",
            "available_hours_per_week": 25,
            "deadline_weeks": 24,
            "user_profile": {
                "learning_style": "project-based learning",
                "experience": "no programming background",
                "motivation": "very high - starting new career",
                "resources": ["laptop", "internet", "coding bootcamp"],
                "preferred_pace": "intensive but sustainable"
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🚀 Demo Scenario {i}: {scenario['name']}")
        print("-" * 40)
        
        try:
            result = planner.generate_study_plan(
                learning_goal=scenario["learning_goal"],
                current_level=scenario["current_level"],
                time_constraint=scenario["time_constraint"],
                available_hours_per_week=scenario["available_hours_per_week"],
                deadline_weeks=scenario["deadline_weeks"],
                user_profile=scenario["user_profile"]
            )
            
            if result["status"] == "success":
                plan = result["final_plan"]
                
                print(f"✅ Successfully generated: {plan.title}")
                print(f"📊 Validation Score: {plan.validation_score}/100")
                print(f"⏰ Duration: {plan.total_duration_weeks} weeks")
                print(f"📚 Topics: {len(plan.topics)} modules")
                print(f"💡 Critic Feedback: {len(plan.critic_feedback)} recommendations")
                
                # Show first few topics
                print("\n🎯 Sample Topics:")
                for j, topic in enumerate(plan.topics[:3], 1):
                    print(f"  {j}. {topic.name} ({topic.estimated_hours}h)")
                
                # Show critic verdict
                critique = result["agent_results"]["critique"]["critique"]
                print(f"\n🔍 Critic Verdict: {critique.get('final_verdict', 'N/A')}")
                
            else:
                print(f"❌ Failed to generate plan: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during scenario {i}: {str(e)}")
        
        print("\n" + "="*50)

def quick_test():
    """Quick test to verify the system works"""
    print("🧪 Quick System Test")
    print("-" * 20)
    
    try:
        # Use working OpenRouter setup
        api_key = "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"
        llm_client = OpenRouterClient(api_key=api_key)
        planner = MultiAgentStudyPlanner(llm_client)
        
        # Minimal test
        result = planner.generate_study_plan(
            learning_goal="Learn basic Python programming",
            current_level="Complete beginner",
            time_constraint="1 month",
            available_hours_per_week=10,
            deadline_weeks=4
        )
        
        if result["status"] == "success":
            print("✅ System working correctly!")
            plan = result["final_plan"]
            print(f"Generated plan: {plan.title}")
            print(f"Validation Score: {plan.validation_score}/100")
            print(f"Topics: {len(plan.topics)} modules")
            print(f"Duration: {plan.total_duration_weeks} weeks")
        else:
            print("❌ System test failed")
            print(f"Error: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Test error: {str(e)}")

if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Quick test")
    print("2. Full demo scenarios")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        quick_test()
    elif choice == "2":
        demo_study_planner()
    else:
        print("Running quick test by default...")
        quick_test()
