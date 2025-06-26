from box import Box
from src.tasks.tokenization_instruction.orchestrator import TokenizationInstructionOrchestrator


def execute(config: Box):
    orchestrator = TokenizationInstructionOrchestrator(config)
    return orchestrator.execute()