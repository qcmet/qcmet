"""exceptions.py.

This module defines custom exceptions used in the QCMet framework.

"""


class MeasurementOutcomesExistError(Exception):
    """Raised when measurement outcomes already exist for the given circuits.

    This is typically used to prevent accidental re-execution or
    overwriting of previously measured quantum circuits in a
    benchmarking workflow.
    """

    def __init__(self, message, errors):
        """Initialize the MeasurementOutcomesExistError.

        Args:
            message (str): A message describing the context of the error.
            errors (Any): Additional error details or metadata.

        """
        # Call the base class constructor with the parameters it needs
        super().__init__(
            f"Measurement outcomes already exist for these circuits: {message}"
        )

        self.errors = errors
