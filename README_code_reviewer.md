# Multi-Agent AI Python Code Reviewer

A sophisticated AI system that reviews beginner-level Python code through collaborative multi-agent architecture.

## 🤖 System Architecture

The system uses three specialized agents that work together:

### 1. Code Reader Agent
- **Role**: Explains code clearly for beginners
- **Function**: Provides jargon-free explanations and concept analysis
- **Output**: Structured code explanations with complexity levels

### 2. Bug Finder Agent  
- **Role**: Detects simple mistakes and issues
- **Function**: Identifies syntax errors, logic issues, and common problems
- **Output**: Detailed issue reports with severity ratings and fixes

### 3. Suggestion Agent
- **Role**: Proposes improvements and best practices
- **Function**: Suggests style, performance, and readability improvements
- **Output**: Actionable suggestions with code examples and benefits

## 🌐 Web Demo

### Quick Demo (No Server Required)

Open the interactive demo directly in your browser:

```bash
# Method 1: Double-click the file
code_review_demo.html

# Method 2: Open with browser
start code_review_demo.html  # Windows
open code_review_demo.html   # macOS
xdg-open code_review_demo.html  # Linux
```

**Demo Features:**
- 🤖 Interactive multi-agent visualization
- 📋 Real-time code review simulation
- 🎯 Agent workflow animation
- 📊 Dynamic quality scoring based on code analysis
- 📱 Mobile-responsive design

### Full Web App (Flask)

For the complete multi-agent system with real AI:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
python code_review_app.py

# Open browser to
http://localhost:5000
```

**Full App Features:**
- Real AI agent collaboration
- Live OpenRouter API integration
- Custom Python code review
- Detailed issue detection and suggestions
- Agent status tracking

## 🚀 Features

- **Multi-Agent Collaboration**: Agents work sequentially to analyze code
- **Flexible LLM Support**: Works with local Ollama or cloud APIs (Groq, OpenRouter, etc.)
- **Comprehensive Code Analysis**: Explains code, finds bugs, suggests improvements
- **Beginner-Friendly**: Jargon-free explanations and clear feedback
- **Web Demo**: Interactive browser-based demo (no server required)
- **Full Web App**: Complete Flask application with real AI integration

## 📦 Installation

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/Eya-ben-nhila/AI-Study-Planner
cd AI-Study-Planner
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

#### Option 2: Cloud API (OpenRouter)
```bash
# Sign up at https://openrouter.ai
# Get your API key and set it in the code
```

## 🎯 Usage

### Basic Example
```python
from multi_agent_code_reviewer import MultiAgentCodeReviewer, OpenRouterClient

# Initialize LLM client
llm_client = OpenRouterClient(api_key="your-api-key")

# Create reviewer
reviewer = MultiAgentCodeReviewer(llm_client)

# Review Python code
code = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    average = total / len(numbers)
    return average
'''

result = reviewer.review_code(code, "calculator.py")

# Access the review results
if result["status"] == "success":
    review = result["review"]
    print(f"File: {review.file_name}")
    print(f"Quality Score: {review.overall_score}/100")
    print(f"Issues Found: {len(review.issues)}")
    print(f"Suggestions: {len(review.suggestions)}")
```

## 📊 Output Structure

The system generates comprehensive code reviews with:

```python
CodeReviewResult(
    file_name="calculator.py",
    code_summary="A function that calculates the average of numbers",
    explanations=[
        CodeExplanation(
            section="calculate_average function",
            explanation="This function computes the arithmetic mean...",
            complexity="beginner",
            key_concepts=["functions", "loops", "arithmetic"]
        )
    ],
    issues=[
        CodeIssue(
            line_number=5,
            issue_type="type_error",
            description="Type conversion issue in string concatenation",
            severity="medium",
            suggestion="Use f-string formatting"
        )
    ],
    suggestions=[
        CodeSuggestion(
            category="performance",
            suggestion="Use built-in sum() function",
            example="total = sum(numbers)",
            benefit="More efficient and Pythonic"
        )
    ],
    overall_score=85,
    feedback_summary=["Found 1 issues", "Generated 2 suggestions"]
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

#### OpenRouter (Cloud)
```python
llm_client = OpenRouterClient(
    api_key="your-openrouter-api-key",
    model="meta-llama/llama-3.1-8b-instruct"
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

1. **Code Reader Agent** analyzes and explains code structure
2. **Bug Finder Agent** scans for issues and problems
3. **Suggestion Agent** proposes improvements and best practices
4. **System** synthesizes results into comprehensive review

Each agent builds upon the previous agent's work, creating increasingly detailed and helpful feedback.

## 🚨 Error Handling

The system includes robust error handling:
- JSON parsing validation for LLM responses
- Fallback responses for API failures
- Graceful degradation when agents encounter issues
- Detailed error reporting for debugging

## 🎯 Example Use Cases

- **Learning Python**: Get explanations of code concepts
- **Code Quality Check**: Find bugs and issues in your code
- **Best Practices**: Learn Python idioms and improvements
- **Educational Settings**: Help students understand code better
- **Code Reviews**: Automated feedback for pull requests

## 🤝 Contributing

This system demonstrates advanced AI agent architecture. Key areas for enhancement:
- Additional specialized agents (Security Agent, Performance Agent)
- Integration with IDEs and code editors
- Support for other programming languages
- Real-time collaborative code reviews

## 📄 License

MIT License - feel free to use and modify for your projects!

---

**Built with ❤️ using multi-agent AI architecture**
