"""
Flask Web App for Multi-Agent Python Code Reviewer
"""

from flask import Flask, render_template, request, jsonify
from multi_agent_code_reviewer import MultiAgentCodeReviewer, OpenRouterClient
import json
import os

app = Flask(__name__)

# Initialize the reviewer with OpenRouter
api_key = "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"
llm_client = OpenRouterClient(api_key=api_key)
reviewer = MultiAgentCodeReviewer(llm_client)

@app.route('/')
def index():
    return render_template('code_review.html')

@app.route('/review_code', methods=['POST'])
def review_code():
    try:
        # Get form data with validation
        code = request.form.get('code', '').strip()
        file_name = request.form.get('file_name', '').strip()
        
        # Validate code input
        if not code:
            return jsonify({
                "success": False,
                "error": "Please provide Python code to review."
            })
        
        if not file_name:
            file_name = "submitted_code.py"
        elif not file_name.endswith('.py'):
            file_name += '.py'
        
        # Generate code review
        result = reviewer.review_code(code, file_name)
        
        if result["status"] == "success":
            review = result["review"]
            
            # Prepare response data
            response_data = {
                "success": True,
                "review": {
                    "file_name": review.file_name,
                    "code_summary": review.code_summary,
                    "overall_score": review.overall_score,
                    "explanations": [
                        {
                            "section": exp.section,
                            "explanation": exp.explanation,
                            "complexity": exp.complexity,
                            "key_concepts": exp.key_concepts
                        }
                        for exp in review.explanations
                    ],
                    "issues": [
                        {
                            "line_number": issue.line_number,
                            "issue_type": issue.issue_type,
                            "description": issue.description,
                            "severity": issue.severity,
                            "suggestion": issue.suggestion
                        }
                        for issue in review.issues
                    ],
                    "suggestions": [
                        {
                            "category": suggestion.category,
                            "suggestion": suggestion.suggestion,
                            "example": suggestion.example,
                            "benefit": suggestion.benefit
                        }
                        for suggestion in review.suggestions
                    ],
                    "feedback_summary": review.feedback_summary
                },
                "agent_results": {
                    "code_reader": result["agent_results"]["code_reader"]["status"],
                    "bug_finder": result["agent_results"]["bug_finder"]["status"],
                    "suggestion": result["agent_results"]["suggestion"]["status"]
                },
                "metadata": {
                    "reviewed_at": result["metadata"]["reviewed_at"],
                    "file_name": result["metadata"]["file_name"],
                    "code_length": result["metadata"]["code_length"]
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
    return jsonify({"status": "healthy", "service": "Multi-Agent Code Reviewer"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
