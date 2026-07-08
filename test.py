
from src.config import get_config_manager
from src.document_processor import DocumentProcessor

config_manager = get_config_manager("config.yaml")
config = config_manager.get_config()

processor = DocumentProcessor(config)
documents = processor.process_knowledge_base()

print(documents)
print(type(documents[0]))
