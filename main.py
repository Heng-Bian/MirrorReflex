import yaml
import re
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- 1. 加载 Prompt 配置文件 ---
with open("prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)
with open("inputs.yaml", "r", encoding="utf-8") as f:
    INPUTS = yaml.safe_load(f)
llm = ChatOpenAI(model="YOUR_MODEL_NAME",
                 temperature=0.7,
                 base_url="YOUR_BASE_URL",
                 api_key='YOUR_API_KEY')
# --- 2. 定义状态 (State) ---
class AgentState(TypedDict):
    # 存储样文和目标内容
    sample_text: str
    target_content: str
    # 存储智能体之间的中间产物
    style_report: str
    story_elements: str
    draft: str
    critic_feedback: str
    # 迭代控制
    iterations: int
    current_score: float
    best_score: float
    best_draft: str

# --- 3. 初始化模型 ---


# --- Helper: 渲染 Prompt ---
def render_prompt(template_key: str, state: AgentState) -> str:
    """
    从配置中读取模板，并使用 state 中的数据填充占位符。
    支持缺失键填充默认值。
    """
    template = PROMPTS.get(template_key, "")
    # 提供默认值以防 state 中缺少某些键
    defaults = {
        "sample_text": "",
        "target_content": "",
        "style_report": "暂无分析报告",
        "story_elements": "暂无故事要素",
        "draft": "暂无初稿",
        "critic_feedback": "初始环节，暂无反馈",
        "iterations": 0
    }
    # 合并默认值和当前状态
    full_data = {**defaults, **state}
    return template.format(**full_data)

# --- 4. 定义智能体节点 (Nodes) ---

def analyst_node(state: AgentState):
    """分析样文的文风指纹及故事要素"""
    prompt = render_prompt('style_analyst', state)
    response = llm.invoke(prompt).content
    
    # 尝试从分析报告中提取“故事要素清单”部分
    elements_match = re.search(r"【故事要素清单】：(.*?)(?=\n\n|\n【|$)", response, re.S)
    story_elements = elements_match.group(1).strip() if elements_match else "未提取到明确的故事要素"
    
    return {
        "style_report": response,
        "story_elements": story_elements
    }

def expander_node(state: AgentState):
    """
    通过大模型判断用户输入是否为指令或过短，并决定是否进行内容扩充。
    """
    # 1. 调用大模型进行意图与内容分析
    analysis_prompt = render_prompt('input_analyzer', state)
    decision = llm.invoke(analysis_prompt).content.strip()
    
    # 2. 根据判断结果决定是否执行拓展
    if "[EXPAND]" in decision:
        print(f"--- 大模型判定：需要扩充内容 ---")
        prompt = render_prompt('content_expander', state)
        response = llm.invoke(prompt)
        return {"target_content": response.content}
    else:
        print(f"--- 大模型判定：保留原始输入 ---")
        return {"target_content": state.get("target_content", "")}

def writer_node(state: AgentState):
    """根据指纹和反馈进行写作"""
    prompt = render_prompt('style_mimic', state)
    response = llm.invoke(prompt)
    return {"draft": response.content, "iterations": state.get('iterations', 0) + 1}

def critic_node(state: AgentState):
    """评判生成内容与样文的相似度"""
    prompt = render_prompt('style_critic', state)
    response = llm.invoke(prompt)
    feedback = response.content
    
    # 解析评分
    score_match = re.search(r"【评分】：\s*(\d+(\.\d+)?)", feedback)
    current_score = float(score_match.group(1)) if score_match else 0.0
    
    best_score = state.get("best_score", 0.0)
    
    # 如果当前更好，记录最佳结果
    if current_score >= best_score:
        return {
            "critic_feedback": feedback,
            "current_score": current_score,
            "best_score": current_score,
            "best_draft": state.get("draft", "")
        }
    
    return {"critic_feedback": feedback, "current_score": current_score}

# --- 5. 定义循环逻辑 (Routing) ---

def should_continue(state: AgentState):
    """判断是继续优化还是结束"""
    # 1. 检查分数是否下降
    if state.get("current_score", 0.0) < state.get("best_score", 0.0):
        print(f"--- 触发降级保护：当前评分 {state['current_score']} 低于历史最佳 {state['best_score']}，立刻回滚并终止 ---")
        return "end"

    # 2. 检查是否达到最大迭代次数
    if state['iterations'] >= 5:
        return "end"
    
    # 3. 检查批评反馈中的最终判定
    feedback = state.get('critic_feedback', "")
    if "[完美交付]" in feedback:
        return "end"
        
    return "continue"

# --- 6. 构建图 (Workflow) ---

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("analyst", analyst_node)
workflow.add_node("expander", expander_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)

# 设置入口
workflow.set_entry_point("analyst")

# 连接逻辑
workflow.add_edge("analyst", "expander")
workflow.add_edge("expander", "writer")
workflow.add_edge("writer", "critic")

# 设置条件分支（循环关键）
workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "continue": "writer",
        "end": END
    }
)

# 编译应用
app = workflow.compile()

# --- 7. 运行示例 ---
inputs = {
    "sample_text": INPUTS.get('sample', ""), 
    "target_content": INPUTS.get('target', ""),
    "iterations": 0,
    "current_score": 0.0,
    "best_score": 0.0,
    "best_draft": ""
    }

# 记录完整状态，避免重复调用 invoke
state_accumulator = inputs.copy()

print("="*50)
print("正在启动文风仿写 Agent 任务流...")

for output in app.stream(inputs):
    for key, value in output.items():
        # 更新累积状态，这样最后一次迭代的结果会被保留
        state_accumulator.update(value)
        
        print(f"--- Node: {key} ---")
        if key == "critic":
            print(f"Current Score: {value.get('current_score')}")
        elif key == "expander":
            # 只有在 expander 节点确实改变了 target_content 时才提示
            if value.get("target_content") != inputs["target_content"]:
                print("内容已完成扩充。")
            else:
                print("输入内容充实，跳过扩充。")
        elif key == "writer":
            print("稿件已生成/更新。")

print("\n" + "="*50)
print("【最终交付结果】")
# 直接从累积的状态中提取 best_draft，不再重复运行
print(state_accumulator.get("best_draft", "未生成有效结果"))
print(f"最终判定分数 (最高): {state_accumulator.get('best_score')}")
print("="*50)
