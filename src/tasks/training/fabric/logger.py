"""
This module defines a custom logger creation function that builds on CSVLogger.
It overrides the experiment.save method to merge CSV log entries by their 'step' key.
"""


from lightning.fabric.loggers import CSVLogger
from typing import Any, Dict, Type, TypeVar, List
from types import MethodType

T = TypeVar("T")


def step_csv_logger(*args: Any, cls: Type[T] = CSVLogger, **kwargs: Any) -> T:
    """
    Create a customized logger instance with an overridden experiment.save method.

    This function instantiates a logger (by default a CSVLogger) using provided positional
    and keyword arguments. It then replaces the logger's experiment.save method with a
    custom implementation that merges CSV log rows by a specified key, 'step'.

    Args:
        *args: Positional arguments to pass to the CSVLogger (or a subclass) constructor.
        cls (Type[T], optional): The logger class to instantiate. Defaults to CSVLogger.
        **kwargs: Keyword arguments to pass to the CSVLogger (or a subclass) constructor.

    Returns:
        T: An instance of the logger with the customized 'save' method.

    Note:
        The customized 'save' method leverages a helper function `merge_by` to combine logs
        sharing the same step number before writing them to a CSV file.
    """
    logger = cls(*args, **kwargs)

    def merge_by(dicts: List[Dict], key: str) -> List[Dict]:
        """
        Merge a list of dictionaries by a common key.

        This helper function iterates over a list of dictionaries and, for each dictionary
        that contains the specified key, merges dictionaries that share the same key value.
        The result is returned as a list of merged dictionaries, sorted by the key's value.

        Args:
            dicts (List[Dict]): A list of dictionaries to be merged.
            key (str): The key used to merge dictionaries (e.g., 'step').

        Returns:
            List[Dict]: A sorted list of merged dictionaries based on the specified key.
        """
        
        from collections import defaultdict

        out = defaultdict(dict)
        for d in dicts:
            if key in d:
                out[d[key]].update(d)
        return [v for _, v in sorted(out.items())]

    def save(self) -> None:
        """
        Customized save method for the experiment.

        This method overrides the default experiment.save behavior. It merges the logged
        metrics based on the 'step' key, determines the complete set of CSV columns from
        the merged data, and then writes the resulting list of dictionaries to the CSV file.
        
        The method uses the csv module to output the header and rows accordingly.

        Args:
            self: This parameter refers to the instance of the experiment associated with
                  the logger.

        Returns:
            None.
        """
        import csv

        if not self.metrics:
            return
        metrics = merge_by(self.metrics, "step")
        keys = sorted({k for m in metrics for k in m})
        with self._fs.open(self.metrics_file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics)

    logger.experiment.save = MethodType(save, logger.experiment)

    return logger