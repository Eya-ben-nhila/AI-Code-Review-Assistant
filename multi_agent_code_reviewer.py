"""
Multi-Agent AI System for Python Code Review

This system uses three specialized agents:
1. Code Reader Agent - Explains code clearly for beginners
2. Bug Finder Agent - Detects simple mistakes and issues
3. Suggestion Agent - Proposes improvements and best practices

The system analyzes Python code and provides comprehensive feedback for beginners.
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
from abc import ABC, abstractmethod


@dataclass
class CodeIssue:
    """Represents a code issue or bug"""
    line_number: int
    issue_type: str
    description: str
    severity: str  # low, medium, high
    suggestion: str


@dataclass
class CodeExplanation:
    """Represents code explanation"""
    section: str
    explanation: str
    complexity: str  # beginner, intermediate, advanced
    key_concepts: List[str]


@dataclass
class CodeSuggestion:
    """Represents improvement suggestion"""
    category: str  # style, performance, readability, best_practice
    suggestion: str
    example: str
    benefit: str


@dataclass
class CodeReviewResult:
    """Complete code review result"""
    file_name: str
    code_summary: str
    explanations: List[CodeExplanation]
    issues: List[CodeIssue]
    suggestions: List[CodeSuggestion]
    overall_score: float
    feedback_summary: List[str]


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
                "HTTP-Referer": "https://github.com/multi-agent-code-reviewer",
                "X-Title": "Multi-Agent Code Reviewer"
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


class CodeReaderAgent(Agent):
    """Explains Python code clearly for beginners"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__("Code Reader Agent", llm_client)
    
    def process(self, code: str, file_name: str = "unknown.py") -> Dict[str, Any]:
        """Generate clear explanations of the code"""
        
        system_prompt = """You are an expert programming teacher who explains Python code to beginners. Your explanations should be:
        1. Clear and simple, avoiding jargon
        2. Structured in logical sections
        3. Focused on what each part does and why
        4. Include key concepts beginners should know
        5. Break down complex ideas into simple steps
        
        Always respond with valid JSON format."""
        
        prompt = f"""
        Explain this Python code for a beginner programmer:
        
        File: {file_name}
        
        Code:
        ```python
        {code}
        ```
        
        Please provide:
        1. Overall summary of what the code does
        2. Section-by-section explanations
        3. Key programming concepts used
        4. Complexity level for each section
        
        Format as JSON:
        {{
            "summary": "Overall description of what this code does",
            "explanations": [
                {{
                    "section": "Section name or line range",
                    "explanation": "Clear explanation for beginners",
                    "complexity": "beginner/intermediate/advanced",
                    "key_concepts": ["concept1", "concept2"]
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
            explanation_data = json.loads(json_content)
            return {
                "agent": self.name,
                "status": "success",
                "explanation": explanation_data,
                "raw_response": response
            }
        except json.JSONDecodeError:
            return {
                "agent": self.name,
                "status": "error",
                "message": "Failed to parse explanation response",
                "raw_response": response
            }


class BugFinderAgent(Agent):
    """Detects simple mistakes and issues in Python code"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__("Bug Finder Agent", llm_client)
    
    def process(self, code: str, file_name: str = "unknown.py") -> Dict[str, Any]:
        """Find bugs and issues in the code"""
        
        # First, check for syntax errors using Python's compiler
        syntax_errors = self._check_syntax_errors(code)
        
        system_prompt = """You are a code quality expert who finds bugs and issues in Python code. Focus on:
        1. Logic errors and common mistakes
        2. Poor coding practices that could cause issues
        3. Security vulnerabilities
        4. Performance issues
        5. Runtime errors that might occur
        6. Code style and readability issues
        
        Rate issues by severity: low, medium, high
        Always respond with valid JSON format."""
        
        prompt = f"""
        Find bugs and issues in this Python code:
        
        File: {file_name}
        
        Code:
        ```python
        {code}
        ```
        
        Please identify:
        1. Line numbers where issues occur
        2. Type of issue (logic_error, runtime_error, security, performance, style, etc.)
        3. Description of the problem
        4. Severity level (low, medium, high)
        5. Suggestions to fix each issue
        
        Format as JSON:
        {{
            "issues": [
                {{
                    "line_number": 10,
                    "issue_type": "logic_error",
                    "description": "What the problem is",
                    "severity": "medium",
                    "suggestion": "How to fix it"
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
            bug_data = json.loads(json_content)
            
            # Combine syntax errors with AI-detected issues
            all_issues = syntax_errors + bug_data.get("issues", [])
            
            return {
                "agent": self.name,
                "status": "success",
                "bugs": {"issues": all_issues},
                "raw_response": response
            }
        except json.JSONDecodeError:
            return {
                "agent": self.name,
                "status": "error",
                "message": "Failed to parse bug detection response",
                "raw_response": response
            }
    
    def _check_syntax_errors(self, code: str) -> List[Dict[str, Any]]:
        """Check for Python syntax errors using the built-in compiler"""
        import ast
        import sys
        
        syntax_errors = []
        lines = code.split('\n')
        
        try:
            # Try to parse the code as AST
            ast.parse(code)
        except SyntaxError as e:
            # Handle syntax errors
            error_info = {
                "line_number": e.lineno or 1,
                "issue_type": "syntax_error",
                "description": f"Syntax Error: {e.msg}",
                "severity": "high",
                "suggestion": f"Fix the syntax error: {e.msg}"
            }
            syntax_errors.append(error_info)
        
        # Check for common syntax issues manually
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for unclosed brackets/parentheses
            if stripped and not stripped.startswith('#'):
                # Count brackets and parentheses
                open_parens = line.count('(') - line.count(')')
                open_brackets = line.count('[') - line.count(']')
                open_braces = line.count('{') - line.count('}')
                
                if open_parens > 0:
                    syntax_errors.append({
                        "line_number": i,
                        "issue_type": "syntax_error",
                        "description": f"Unclosed parenthesis on line {i}",
                        "severity": "high",
                        "suggestion": "Add missing closing parenthesis ')'"
                    })
                
                if open_brackets > 0:
                    syntax_errors.append({
                        "line_number": i,
                        "issue_type": "syntax_error",
                        "description": f"Unclosed bracket on line {i}",
                        "severity": "high",
                        "suggestion": "Add missing closing bracket ']'"
                    })
                
                if open_braces > 0:
                    syntax_errors.append({
                        "line_number": i,
                        "issue_type": "syntax_error",
                        "description": f"Unclosed brace on line {i}",
                        "severity": "high",
                        "suggestion": "Add missing closing brace '}'"
                    })
                
                # Check for invalid indentation (common Python issue)
                if line.startswith('\t') and ' ' in line:
                    syntax_errors.append({
                        "line_number": i,
                        "issue_type": "syntax_error",
                        "description": "Mixed tabs and spaces in indentation",
                        "severity": "high",
                        "suggestion": "Use either tabs or spaces consistently for indentation"
                    })
                
                # Check for common syntax mistakes
                if stripped.endswith(':') and not any(keyword in stripped for keyword in ['if', 'elif', 'else', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'with']):
                    syntax_errors.append({
                        "line_number": i,
                        "issue_type": "syntax_error",
                        "description": "Colon used without valid control structure",
                        "severity": "medium",
                        "suggestion": "Remove colon or use with proper control structure (if, for, def, etc.)"
                    })
                
                # Check for invalid characters
                try:
                    line.encode('ascii')
                except UnicodeEncodeError:
                    syntax_errors.append({
                        "line_number": i,
                        "issue_type": "syntax_error",
                        "description": "Non-ASCII characters found in code",
                        "severity": "low",
                        "suggestion": "Use only ASCII characters or add proper encoding declaration"
                    })
        
        return syntax_errors


class SuggestionAgent(Agent):
    """Proposes improvements and best practices for Python code"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__("Suggestion Agent", llm_client)
    
    def process(self, code: str, file_name: str = "unknown.py") -> Dict[str, Any]:
        """Generate improvement suggestions"""
        
        system_prompt = """You are a senior Python developer who suggests improvements. Focus on:
        1. Code style and readability (PEP 8)
        2. Performance optimizations
        3. Best practices and patterns
        4. Error handling and robustness
        5. Documentation and comments
        
        Provide concrete examples with before/after code.
        Always respond with valid JSON format."""
        
        prompt = f"""
        Suggest improvements for this Python code:
        
        File: {file_name}
        
        Code:
        ```python
        {code}
        ```
        
        Please provide:
        1. Categorized suggestions (style, performance, best_practice, readability)
        2. Specific improvement recommendations
        3. Code examples showing the improvement
        4. Benefits of each suggestion
        
        Format as JSON:
        {{
            "suggestions": [
                {{
                    "category": "style/performance/readability/best_practice",
                    "suggestion": "What to improve",
                    "example": "Code example showing the improvement",
                    "benefit": "Why this is better"
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
            suggestion_data = json.loads(json_content)
            return {
                "agent": self.name,
                "status": "success",
                "suggestions": suggestion_data,
                "raw_response": response
            }
        except json.JSONDecodeError:
            return {
                "agent": self.name,
                "status": "error",
                "message": "Failed to parse suggestion response",
                "raw_response": response
            }


class MultiAgentCodeReviewer:
    """Orchestrates collaboration between all code review agents"""
    
    def __init__(self, llm_client: LLMClient):
        self.code_reader_agent = CodeReaderAgent(llm_client)
        self.bug_finder_agent = BugFinderAgent(llm_client)
        self.suggestion_agent = SuggestionAgent(llm_client)
        self.llm_client = llm_client
    
    def review_code(self, code: str, file_name: str = "unknown.py") -> Dict[str, Any]:
        """Generate comprehensive code review through agent collaboration"""
        
        print("🔍 Starting Multi-Agent Code Review...")
        print(f"📁 File: {file_name}")
        print(f"📝 Code length: {len(code)} characters")
        
        # Step 1: Code Reader Agent explains the code
        print("\n📖 Step 1: Code Reader Agent analyzing code...")
        code_result = self.code_reader_agent.process(code, file_name)
        
        if code_result["status"] != "success":
            return {"error": "Code explanation failed", "details": code_result}
        
        # Step 2: Bug Finder Agent detects issues
        print("🐛 Step 2: Bug Finder Agent scanning for issues...")
        bug_result = self.bug_finder_agent.process(code, file_name)
        
        if bug_result["status"] != "success":
            return {"error": "Bug detection failed", "details": bug_result}
        
        # Step 3: Suggestion Agent proposes improvements
        print("💡 Step 3: Suggestion Agent generating improvements...")
        suggestion_result = self.suggestion_agent.process(code, file_name)
        
        if suggestion_result["status"] != "success":
            return {"error": "Suggestion generation failed", "details": suggestion_result}
        
        # Step 4: Generate final review result
        print("✅ Step 4: Creating comprehensive code review...")
        final_review = self._create_final_review(
            code_result,
            bug_result,
            suggestion_result,
            file_name,
            code
        )
        
        return {
            "status": "success",
            "review": final_review,
            "agent_results": {
                "code_reader": code_result,
                "bug_finder": bug_result,
                "suggestion": suggestion_result
            },
            "metadata": {
                "reviewed_at": datetime.now().isoformat(),
                "file_name": file_name,
                "code_length": len(code)
            }
        }
    
    def _create_final_review(self, 
                           code_result: Dict[str, Any],
                           bug_result: Dict[str, Any], 
                           suggestion_result: Dict[str, Any],
                           file_name: str,
                           code: str) -> CodeReviewResult:
        """Create the final comprehensive code review"""
        
        explanation_data = code_result["explanation"]
        bug_data = bug_result["bugs"]
        suggestion_data = suggestion_result["suggestions"]
        
        # Create CodeExplanation objects
        explanations = []
        for exp in explanation_data.get("explanations", []):
            explanations.append(CodeExplanation(
                section=exp.get("section", ""),
                explanation=exp.get("explanation", ""),
                complexity=exp.get("complexity", "beginner"),
                key_concepts=exp.get("key_concepts", [])
            ))
        
        # Create CodeIssue objects
        issues = []
        for issue in bug_data.get("issues", []):
            issues.append(CodeIssue(
                line_number=issue.get("line_number", 0),
                issue_type=issue.get("issue_type", ""),
                description=issue.get("description", ""),
                severity=issue.get("severity", "low"),
                suggestion=issue.get("suggestion", "")
            ))
        
        # Create CodeSuggestion objects
        suggestions = []
        for suggestion in suggestion_data.get("suggestions", []):
            suggestions.append(CodeSuggestion(
                category=suggestion.get("category", ""),
                suggestion=suggestion.get("suggestion", ""),
                example=suggestion.get("example", ""),
                benefit=suggestion.get("benefit", "")
            ))
        
        # Calculate overall score
        high_issues = sum(1 for issue in issues if issue.severity == "high")
        medium_issues = sum(1 for issue in issues if issue.severity == "medium")
        base_score = 100
        score = base_score - (high_issues * 15) - (medium_issues * 5)
        overall_score = max(0, min(100, score))
        
        # Generate feedback summary
        feedback_summary = [
            f"Found {len(issues)} issues ({high_issues} high, {medium_issues} medium)",
            f"Generated {len(suggestions)} improvement suggestions",
            f"Code complexity: {'beginner' if all(exp.complexity == 'beginner' for exp in explanations) else 'intermediate/advanced'}",
            f"Overall quality score: {overall_score}/100"
        ]
        
        return CodeReviewResult(
            file_name=file_name,
            code_summary=explanation_data.get("summary", ""),
            explanations=explanations,
            issues=issues,
            suggestions=suggestions,
            overall_score=overall_score,
            feedback_summary=feedback_summary
        )


def main():
    """Example usage of the multi-agent code reviewer"""
    
    print("🤖 Multi-Agent Python Code Reviewer")
    print("=" * 50)
    
    # Initialize LLM client (choose one)
    # Option 1: Local Ollama
    # llm_client = OllamaClient(model_name="llama3.1:8b")
    
    # Option 2: Cloud OpenRouter
    llm_client = OpenRouterClient(api_key="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")
    
    # Initialize the multi-agent reviewer
    reviewer = MultiAgentCodeReviewer(llm_client)
    
    # Example Python code to review
    sample_code = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    average = total / len(numbers)
    return average

def main():
    scores = [85, 92, 78, 90, 88]
    avg = calculate_average(scores)
    print("Average score is: " + avg)
    
main()
'''
    
    # Generate code review
    result = reviewer.review_code(sample_code, "average_calculator.py")
    
    if result["status"] == "success":
        print("\n🎉 Code Review Completed Successfully!")
        print("\n📋 COMPREHENSIVE CODE REVIEW:")
        print("=" * 40)
        
        review = result["review"]
        print(f"📁 File: {review.file_name}")
        print(f"📝 Summary: {review.code_summary}")
        print(f"📊 Quality Score: {review.overall_score}/100")
        print(f"🐛 Issues Found: {len(review.issues)}")
        print(f"💡 Suggestions: {len(review.suggestions)}")
        
        print("\n📖 Code Explanations:")
        for i, exp in enumerate(review.explanations, 1):
            print(f"{i}. {exp.section} ({exp.complexity})")
            print(f"   📖 {exp.explanation}")
            if exp.key_concepts:
                print(f"   🔑 Concepts: {', '.join(exp.key_concepts)}")
            print()
        
        print("🐛 Issues Found:")
        for issue in review.issues:
            print(f"   Line {issue.line_number}: {issue.issue_type} ({issue.severity})")
            print(f"   📝 {issue.description}")
            print(f"   💡 {issue.suggestion}")
            print()
        
        print("💡 Improvement Suggestions:")
        for suggestion in review.suggestions:
            print(f"   📂 {suggestion.category}")
            print(f"   💡 {suggestion.suggestion}")
            print(f"   📝 {suggestion.example}")
            print(f"   ✅ {suggestion.benefit}")
            print()
        
        print(f"\n📊 Summary:")
        for feedback in review.feedback_summary:
            print(f"   • {feedback}")
        
        print(f"\n🤖 Reviewed at: {result['metadata']['reviewed_at']}")
        
    else:
        print("❌ Error generating code review:")
        print(result.get("error", "Unknown error"))


if __name__ == "__main__":
    main()
