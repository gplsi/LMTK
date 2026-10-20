from src.tasks.convert.formats.ddp_huggingface import DDPtoHuggingFace
from src.tasks.convert.formats.fsdp_huggingface import FSDPtoHuggingFace

FORMAT_HANDLERS = {
    "fsdp_huggingface": FSDPtoHuggingFace,
    "ddp_huggingface": DDPtoHuggingFace,
}
