"""Cheap preset — GLM planner + Composer workers (all via Cursor SDK)."""

from hybrid_agent import HybridAgent

if __name__ == "__main__":
    cheap_agent = HybridAgent(
        advisor_model="glm-5.2",
        orchestrator_model="glm-5.2",
        worker_model="composer-2.5",
        verbose=True,
    )
    result = cheap_agent.process("Create a marketing strategy for my SaaS product")
    print("\n--- Result ---\n")
    print(result)
