import os
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.llms import ChatOpenAI
from flask import Flask, request, jsonify

class SOWState(TypedDict):
    json_data: dict         
    draft: str              
    compliance: str         
    formatted: str          
    messages: List[dict]    

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4", temperature=0.7)

def drafting_agent(state: SOWState) -> SOWState:
    try:
        data = state.get("json_data", {})
        project_name = data.get("project_name")
        services = data.get("services_description")
        if not project_name or not services:
            draft_text = "Error: Mandatory fields 'project_name' and/or 'services_description' are missing."
        else:
            deliverables = data.get("deliverables", [])
            milestones = data.get("milestones", [])
            draft_text = f"Draft SOW for {project_name}:\nScope: {services}\n"
            if deliverables:
                draft_text += "Deliverables:\n" + "\n".join(f"- {d}" for d in deliverables) + "\n"
            if milestones:
                draft_text += "Milestones:\n" + "\n".join(
                    f"- {m.get('milestone', 'N/A')}: {m.get('date', 'N/A')}" for m in milestones
                ) + "\n"
        state["draft"] = draft_text
        state.setdefault("messages", []).append({
            "role": "assistant",
            "agent": "Drafting",
            "content": draft_text
        })
except Exception as e:
        error_text = f"Error in Drafting Agent: {str(e)}"
        state.setdefault("messages", []).append({"role": "error", "agent": "Drafting", "content": error_text})
    return state

def compliance_agent(state: SOWState) -> SOWState:
    try:
        draft = state.get("draft", "")
        if "Error:" in draft:
            compliance_feedback = "Compliance check skipped due to draft errors."
        else:
            prompt = (
                "Review the following SOW draft for compliance with legal and contractual requirements. "
                "Identify any issues or missing elements:\n\n" + draft
            )
            compliance_feedback = llm.invoke(prompt)
        state["compliance"] = compliance_feedback
        state.setdefault("messages", []).append({
            "role": "assistant",
            "agent": "Compliance",
            "content": compliance_feedback
        })
    except Exception as e:
        error_text = f"Error in Compliance Agent: {str(e)}"
        state.setdefault("messages", []).append({"role": "error", "agent": "Compliance", "content": error_text})
    return state

def formatting_agent(state: SOWState) -> SOWState:
    try:
        draft = state.get("draft", "")
        compliance_feedback = state.get("compliance", "")
        if "Error:" in draft:
            formatted_text = draft
        else:
            prompt = (
                "Using the following draft and compliance feedback, produce a final, professionally formatted "
                "Statement of Work (SOW) document. Include clear headings and bullet points for deliverables and milestones.\n\n"
                "Draft:\n" + draft + "\n\nCompliance Feedback:\n" + compliance_feedback
            )
            formatted_text = llm.invoke(prompt)
        state["formatted"] = formatted_text
        state.setdefault("messages", []).append({
            "role": "assistant",
            "agent": "Formatting",
            "content": formatted_text
        })
    except Exception as e:
        error_text = f"Error in Formatting Agent: {str(e)}"
        state.setdefault("messages", []).append({"role": "error", "agent": "Formatting", "content": error_text})
    return state

# LangGraph Workflow
workflow = StateGraph(SOWState)
workflow.add_node("Drafting", drafting_agent)
workflow.add_node("Compliance", compliance_agent)
workflow.add_node("Formatting", formatting_agent)

workflow.add_edge(START, "Drafting")
workflow.add_edge("Drafting", "Compliance")
workflow.add_edge("Compliance", "Formatting")
workflow.add_edge("Formatting", END)

agentic_app = workflow.compile()

# Flask ENDPOINT
app = Flask(__name__)

@app.route('/generate_sow', methods=['POST'])
def generate_sow():
    data = request.get_json()
    initial_state: SOWState = {
        "json_data": data.get("json_data", {}),
        "draft": "",
        "compliance": "",
        "formatted": "",
        "messages": []
    }
    try:
        final_state = agentic_app.invoke(initial_state)
        return jsonify({
            "formatted_sow": final_state.get("formatted"),
            "messages": final_state.get("messages")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)


