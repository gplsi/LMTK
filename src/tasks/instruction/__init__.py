from box import Box
from src.tasks.instruction.orchestrator import InstructionOrchestrator

# src/tasks/instruction.py
def execute(config: Box):
    orchestrator = InstructionOrchestrator(config)
    return orchestrator.execute()
