"""
基于经验推广框架的用户交互 Agent —— 智能对话模式

像聊天一样自然交互，自动识别用户意图：
- learn：用户讲述事件 → 存入知识库
- learn_file：用户要从文件学习
- predict：用户寻求预测/建议
- chat：日常对话
"""

from 经验推广框架 import ExperienceAgent, LLM

CONVERSATION_PROMPT = """你是一个经验推理助手，基于经验推广框架运行。

## 你的能力
1. 学习经验：用户讲述亲身经历的事件时，你会自然地回应，系统会在后台存入知识库
2. 预测推理：用户问"如果...会怎样"、"遇到...会如何"时，系统会调用推理引擎做预测
3. 日常对话：闲聊、回答知识库相关问题等

## 知识库状态
- 已学习经验数：{exp_count}
- 知识库实体数：{entity_count}

请像一位有经验的伙伴一样自然地和用户对话。"""

CLASSIFY_PROMPT = """根据用户输入判断意图，只输出一个词：
learn — 用户在讲述一件具体的事件/经历，可以存入知识库
learn_file — 用户想从文件中学习经验（提到"文件"、"文档"、"md"、"txt"、"读"文件等）
predict — 用户寻求建议、预测、判断（"怎么办"、"...会怎样"、"该怎么做"、"建议"、"如何应对"等假设或建议性问题）
chat — 其他：问候、闲聊、提问知识库内容等

知识库：{exp_count} 条经验，{entity_count} 个实体
用户：{text}"""


def _classify(text: str, exp_count: int, entity_count: int, llm: LLM) -> str:
    resp = llm.ask([{"role": "user", "content": CLASSIFY_PROMPT.format(
        text=text, exp_count=exp_count, entity_count=entity_count)}]).strip().lower()
    if "learn_file" in resp:
        return "learn_file"
    if "learn" in resp:
        return "learn"
    if "predict" in resp:
        return "predict"
    return "chat"


def main():
    llm = LLM()
    agent = ExperienceAgent(llm)
    messages = []

    print("=" * 56)
    print("  经验推广框架 —— 智能对话")
    print("  像聊天一样输入经验和提问")
    print("=" * 56)
    print()

    while True:
        try:
            raw = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("退出")
            break

        state = dict(exp_count=len(agent.experiences), entity_count=len(agent.kb.entities))
        intent = _classify(raw, **state, llm=llm)

        if intent == "learn":
            sys_prompt = CONVERSATION_PROMPT.format(**state)
            full = ""
            for chunk in llm.ask_stream(
                [{"role": "system", "content": sys_prompt}]
                + messages[-8:]
                + [{"role": "user", "content": raw}]
            ):
                print(chunk, end="", flush=True)
                full += chunk
            print("\n  ▶ 学习经验入库中...", end="", flush=True)
            agent.learn(raw)
            print(f"\r  ✓ 已存入经验库（共 {len(agent.experiences)} 条经验，{len(agent.kb.entities)} 个实体）")
            messages.extend([{"role": "user", "content": raw}, {"role": "assistant", "content": full}])

        elif intent == "learn_file":
            extract_prompt = f"""从用户输入中提取要学习的文件路径。只输出路径，不要其他内容。
用户输入：{raw}"""
            filepath = llm.ask([{"role": "user", "content": extract_prompt}]).strip()
            result = agent.learn_from_file(filepath)
            print(f"  {result}")

        elif intent == "predict":
            print()
            full = ""
            for chunk in agent.predict_stream(raw, verbose=True):
                print(chunk, end="", flush=True)
                full += chunk
            print()
            messages.extend([{"role": "user", "content": raw}, {"role": "assistant", "content": full}])

        else:  # chat
            sys_prompt = CONVERSATION_PROMPT.format(**state)
            full = ""
            for chunk in llm.ask_stream(
                [{"role": "system", "content": sys_prompt}]
                + messages[-8:]
                + [{"role": "user", "content": raw}]
            ):
                print(chunk, end="", flush=True)
                full += chunk
            print()
            messages.extend([{"role": "user", "content": raw}, {"role": "assistant", "content": full}])


if __name__ == "__main__":
    main()
