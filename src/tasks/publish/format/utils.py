from src.tasks.publish.format.ddp import ConvertDDPCheckpoint
from src.tasks.publish.format.fsdp import ConvertFSDPCheckpoint

FORMAT_HANDLERS = {
    "fsdp": ConvertFSDPCheckpoint,
    "ddp": ConvertDDPCheckpoint,
}
