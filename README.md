# Multi-Agent AI Study Plan Generator

A sophisticated AI system that generates realistic, validated study plans through collaborative multi-agent architecture.

## 🤖 System Architecture

The system uses three specialized agents that work together:

### 1. Curriculum Agent
- **Role**: Proposes comprehensive learning roadmaps
- **Function**: Creates structured topic sequences with prerequisites and difficulty levels
- **Output**: Detailed curriculum with time estimates for each topic

### 2. Time Estimation Agent  
- **Role**: Validates schedule feasibility
- **Function**: Analyzes time constraints, adjusts estimates, creates weekly schedules
- **Output**: Validation scores and feasible timeline allocations

### 3. Critic Agent
- **Role**: Flags unrealistic plans and potential issues
- **Function**: Identifies knowledge gaps, workload concerns, and practical feasibility
- **Output**: Constructive feedback and improvement recommendations

## 🚀 Features

- **Multi-Agent Collaboration**: Agents work sequentially to refine and validate plans
- **Flexible LLM Support**: Works with local Ollama or cloud APIs (Groq, etc.)
- **Realistic Time Management**: Validates schedules against actual available time
- **Comprehensive Critique**: Identifies potential issues before implementation
- **Structured Output**: Generates detailed, actionable study plans

## 📦 Installation

### 1. Clone and Install Dependencies
```bash
git clone <repository-url>
cd multi-agent-study-planner
pip install -r requirements.txt
```

### 2. Choose Your LLM Setup

#### Option 1: Local Ollama (Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended model
ollama pull llama3.1:8b

# Start Ollama server
ollama serve
```

#### Option 2: Cloud API (Groq)
```bash
# Sign up at https://groq.com
# Get your API key and set it in the code
```

## 🎯 Usage

### Basic Example
```python
from multi_agent_study_planner import MultiAgentStudyPlanner, OllamaClient

# Initialize LLM client
llm_client = OllamaClient(model_name="llama3.1:8b")

# Create planner
planner = MultiAgentStudyPlanner(llm_client)

# Generate study plan
result = planner.generate_study_plan(
    learning_goal="Learn Python for Data Science",
    current_level="Beginner programmer",
    time_constraint="3 months part-time",
    available_hours_per_week=15,
    deadline_weeks=12
)

# Access the validated plan
plan = result["final_plan"]
print(f"Plan: {plan.title}")
print(f"Duration: {plan.total_duration_weeks} weeks")
print(f"Validation Score: {plan.validation_score}/100")
```

### Advanced Usage with User Profile
```python
user_profile = {
    "learning_style": "visual and hands-on",
    "experience": "some programming in JavaScript",
    "motivation": "high - career change",
    "resources": ["laptop", "internet", "online courses"],
    "preferred_pace": "intensive but sustainable"
}

result = planner.generate_study_plan(
    learning_goal="Master Machine Learning",
    current_level="Intermediate Python developer",
    time_constraint="6 months full-time",
    available_hours_per_week=40,
    deadline_weeks=24,
    user_profile=user_profile
)
```

## 📊 Output Structure

The system generates comprehensive study plans with:

```python
StudyPlan(
    title="Complete Python Data Science Path",
    description="Comprehensive curriculum from basics to ML",
    total_duration_weeks=12,
    topics=[
        StudyTopic(
            name="Python Fundamentals",
            description="Core Python programming concepts",
            estimated_hours=25.0,
            prerequisites=[],
            difficulty_level="beginner"
        ),
        # ... more topics
    ],
    weekly_schedule={
        "week_1": ["Python Fundamentals", "Setup & Environment"],
        "week_2": ["Data Structures", "Control Flow"],
        # ... more weeks
    },
    daily_hours=2.5,
    validation_score=87,
    critic_feedback=[
        "Consider adding more practice projects",
        "Schedule looks realistic and achievable"
    ]
)
```

## 🔧 Configuration

### LLM Client Options

#### Ollama (Local)
```python
llm_client = OllamaClient(
    model_name="llama3.1:8b",  # or mistral, qwen2.5, phi3
    base_url="http://localhost:11434"
)
```

#### Groq (Cloud)
```python
llm_client = GroqClient(
    api_key="your-groq-api-key",
    model="llama3-70b-8192"
)
```

### Recommended Models

| Model | Why Use It | Setup |
|-------|------------|-------|
| Llama 3.1 (8B) | Great reasoning, fast | `ollama pull llama3.1:8b` |
| Mistral 7B | Lightweight, stable | `ollama pull mistral` |
| Qwen 2.5 (7B) | Strong instruction following | `ollama pull qwen2.5:7b` |
| Phi-3 Mini | Very fast, low RAM | `ollama pull phi3` |

## 🎭 Agent Collaboration Flow

1. **Curriculum Agent** generates initial learning roadmap
2. **Time Estimation Agent** validates and adjusts time estimates
3. **Critic Agent** reviews for realism and feasibility
4. **System** synthesizes results into final validated plan

Each agent builds upon the previous agent's work, creating increasingly refined and validated output.

## 🚨 Error Handling

The system includes robust error handling:
- JSON parsing validation for LLM responses
- Fallback responses for API failures
- Graceful degradation when agents encounter issues
- Detailed error reporting for debugging

## 🎯 Example Use Cases

- **Career Transition**: "Learn web development for career change"
- **Skill Enhancement**: "Master advanced Python techniques"
- **Academic Preparation**: "Prepare for computer science degree"
- **Hobby Learning**: "Learn game development in spare time"

## 🤝 Contributing

This system demonstrates advanced AI agent architecture. Key areas for enhancement:
- Additional specialized agents (Resource Agent, Progress Tracker)
- Integration with learning platforms
- Visual plan generation
- Progress monitoring and adaptation

## 📄 License

MIT License - feel free to use and modify for your learning projects!

---

**Built with ❤️ using multi-agent AI architecture**
