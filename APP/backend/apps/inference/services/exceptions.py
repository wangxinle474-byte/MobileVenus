class InferenceServiceException(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class InputValidationException(InferenceServiceException):
    pass


class InferenceFailedException(InferenceServiceException):
    pass


class ISPRenderException(InferenceServiceException):
    pass


class StorageException(InferenceServiceException):
    pass
