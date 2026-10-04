class InvalidDocument(ValueError):
    """A document or operation does not satisfy the public contract."""


class NotFound(LookupError):
    """A requested record does not exist."""


class Conflict(RuntimeError):
    """A revision precondition failed."""


class StorageUnavailable(RuntimeError):
    """The durable store could not complete the request."""
