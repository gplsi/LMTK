"""
Anonymization Orchestrator Module

This module provides the AnonymizeOrchestrator class, which handles the orchestration 
of the text anonymization process. The workflow includes validating configuration, 
processing input data to remove sensitive information, and generating anonymized 
outputs for further use or storage.
"""

import os
import subprocess

from box import Box
from src.utils.orchestrator import BaseOrchestrator
from src.utils.logging import get_logger

class AnonymizationOrchestrator(BaseOrchestrator):
    """
    Orchestrates the anonymization workflow.
    """
    
    def __init__(self, config: Box):
        super().__init__(config)
        self.logger = get_logger(__name__, self.verbose_level)

    def _validate_anonymization_config(self):
        """
        Validate the anonymization configuration.
        
        :raises ValueError: If the anonymization configuration is missing or incomplete.
        """
        if not hasattr(self.config, 'anonymization') or not self.config.anonymization:
            raise ValueError("Anonymization configuration must be provided")
        if not hasattr(self.config.anonymization, 'source') or not hasattr(self.config.anonymization.source, 'path'):
            raise ValueError("Anonymization source must be provided")
        if not hasattr(self.config.anonymization, 'output') or not hasattr(self.config.anonymization.output, 'path'):
            raise ValueError("Anonymization output must be provided")
        if not hasattr(self.config.anonymization, 'models_source') or not hasattr(self.config.anonymization.models_source, 'path'):
            raise ValueError("Anonymization models_source must be provided")
        if not hasattr(self.config.anonymization, 'models'):
            raise ValueError("Anonymization models must be provided")
        if not hasattr(self.config.anonymization, 'regexes') or not hasattr(self.config.anonymization.regexes, 'path'):
            raise ValueError("Anonymization regexes must be provided")
        if not hasattr(self.config.anonymization, 'truecaser') or not hasattr(self.config.anonymization.truecaser, 'path'):
            raise ValueError("Anonymization truecaser must be provided")
        if not hasattr(self.config.anonymization, 'format'):
            raise ValueError("Anonymization format must be provided")
        if not hasattr(self.config.anonymization, 'method'):
            raise ValueError("Anonymization method must be provided")
        if not hasattr(self.config.anonymization, 'labels'):
            raise ValueError("Anonymization labels must be provided")
        if hasattr(self.config.anonymization, 'store_original') is None:
            raise ValueError("Anonymization store_original must be provided")
        if hasattr(self.config.anonymization, 'aggregate_output') is None:
            raise ValueError("Anonymization aggregate_output must be provided")
        if hasattr(self.config.anonymization, 'skip_existing') is None:
            raise ValueError("Anonymization skip_existing must be provided")


    def _anonymize_data(self):
        """
        Process the input data to remove sensitive information using a Docker container.
        
        :return: None
        """

        docker_image = self.config.anonymization.get("docker_image", "anonymization:latest")
        input_path = self.config.anonymization.source.path
        output_path = self.config.anonymization.output.path
        models_path = self.config.anonymization.models_source.path
        models = self.config.anonymization.models
        models_ids = [ m.mid for m in models ]
        models_types = [ m.mtype for m in models ]
        regexes_path = self.config.anonymization.regexes.path
        truecaser_path = self.config.anonymization.truecaser.path
        format = self.config.anonymization.format
        method = self.config.anonymization.method
        labels = self.config.anonymization.labels
        store_original = self.config.anonymization.store_original
        aggregate_output = self.config.anonymization.aggregate_output
        skip_existing = self.config.anonymization.skip_existing

        docker_command = [
            "docker", "run", "--rm",
            "-v", f"{os.path.abspath(input_path)}:/home/anonym/input",
            "-v", f"{os.path.abspath(output_path)}:/home/anonym/output",
        ]

        if models_path:
            docker_command.extend(["-v", f"{os.path.abspath(models_path)}:/home/anonym/models"])
        if truecaser_path:
            docker_command.extend(["-v", f"{os.path.abspath(truecaser_path)}:/home/anonym/truecaser"])
        if regexes_path:
            docker_command.extend(["-v", f"{os.path.abspath(regexes_path)}:/home/anonym/regexes"])
        
        docker_command.extend([
            docker_image,
            "-i", f"input/",
            "-o", f"output/",
            "--format", format,
            "--anonym_method", method,
        ])

        for model_id, model_type in zip(models_ids, models_types):
            docker_command.extend(["--models", model_id, "--type_of_models", model_type])
        for label in labels:
            docker_command.extend(["--inline_labels", label])

        if regexes_path:
            docker_command.extend(["--regexes", f"/home/anonym/regexes/{os.path.basename(regexes_path)}"])
        if truecaser_path:
            docker_command.extend(["--truecaser", f"/home/anonym/truecaser/{os.path.basename(truecaser_path)}"])

        if store_original:
            docker_command.append("--store_original")
        if aggregate_output:
            docker_command.append("--aggregate_output")
        if skip_existing:
            docker_command.append("--skip_existing")

        self.logger.info(f"Running Docker command: {' '.join(docker_command)}")
        subprocess.run(docker_command, check=True)
    
    def execute(self):
        """
        Execute the anonymization workflow.
        
        This method orchestrates the validation of configuration, 
        anonymization of data, and saving of the anonymized output.
        """
        try:
            self._validate_anonymization_config()
            
            self.logger.info("Starting anonymization process...")
            self._anonymize_data()
            
            self.logger.info(f"✅ Anonymization workflow completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Error in anonymization workflow: {e}")
            raise