"""Logfire setup for Pydantic AI traces."""

import logfire


_configured = False


def configure_monitoring() -> None:
    """Send traces to Logfire when configured and print spans locally."""
    global _configured
    if _configured:
        return

    logfire.configure(
        console=logfire.ConsoleOptions(span_style="simple"),
        send_to_logfire="if-token-present",
    )
    logfire.instrument_pydantic_ai()
    _configured = True
