from .orchestrator import ConvertOrchestrator
from box import Box

def execute(config: Box):
    orchestrator = ConvertOrchestrator(config)
    return orchestrator.execute()