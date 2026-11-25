from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Pydantic schema representing the health-check response.
    
    Returns standard status information confirming the server and database are functioning.
    """
    status: str = Field(
        default="ok",
        description="Application status flag (e.g., 'ok' or 'degraded')",
        examples=["ok"]
    )
    message: str = Field(
        default="Voice AI Doctor Appointment Assistant Backend is running",
        description="Human-friendly status explanation",
        examples=["Voice AI Doctor Appointment Assistant Backend is running"]
    )
    database: str = Field(
        default="connected",
        description="SQLite database connection status",
        examples=["connected"]
    )
    environment: str = Field(
        default="development",
        description="Active runtime environment",
        examples=["development"]
    )
