"""
基于经验推广框架的快速训练预测 Agent

工作模式（非正常对话）：
  1. 询问用户训练的初始文件路径
  2. 从文件里学习经验
  3. 循环询问用户要预测什么，输出预测到终端
"""

from 经验推广框架 import ExperienceAgent, LLM


def main():
    llm = LLM()
    agent = ExperienceAgent(llm)

    print("=" * 56)
    print("  经验推广框架 —— 快速训练预测 Agent")
    print("  模式：文件学习 → 预测输出")
    print("=" * 56)

    while True:
        filepath = input("\n请输入训练文件路径（输入 quit 退出）: ").strip()
        if not filepath:
            continue
        if filepath.lower() in ("quit", "exit", "q"):
            print("退出")
            break

        result = agent.learn_from_file(filepath)
        print(f"  {result}")

        if "✓" in result:
            break

    print(f"\n知识库就绪：{len(agent.experiences)} 条经验，{len(agent.kb.entities)} 个实体\n")

    while True:
        try:
            query = input("\n请输入要预测的场景（输入 quit 退出）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("退出")
            break

        print()
        for chunk in agent.predict_stream(query, verbose=True):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
