"""Default HybridAgent example — all stages use composer-2.5."""

from hybrid_agent import HybridAgent

if __name__ == "__main__":
    agent = HybridAgent(verbose=True)
    result = agent.process("Write a launch plan for a new AI coding tool")
    print("\n--- Result ---\n")
    print(result)
