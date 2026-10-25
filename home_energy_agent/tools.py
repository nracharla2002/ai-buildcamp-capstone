"""The two small tools used by the Home Energy Helper agent."""

from pydantic import BaseModel, Field

from home_energy_agent.rag import search


class BillPeriod(BaseModel):
    """Details for one user-provided electricity bill period."""

    month: str = Field(description="The bill-period label as the user gave it.")
    kwh: float = Field(ge=0, description="Electricity use in kilowatt-hours.")
    bill_amount: float = Field(ge=0, description="Total bill amount.")
    billing_days: int | None = Field(
        default=None,
        ge=1,
        description="Number of billing days, or null if unknown.",
    )


class HomeEnergyTools:
    """Small, focused tools registered with the PydanticAI agent."""

    def __init__(self, index):
        self.index = index

    def search_energy_guidance(self, question: str) -> list[dict]:
        """Search Energy.gov chunks and return excerpts with citation metadata."""
        print(f"TOOL CALL: search_energy_guidance({question!r})")
        results = search(question, self.index)
        print(f"TOOL RESULT: search_energy_guidance returned {len(results)} records")
        return [
            {
                "title": result.get("title"),
                "topic": result.get("topic"),
                "content": result.get("content"),
                "source_name": result.get("source_name"),
                "source_url": result.get("source_url"),
            }
            for result in results
        ]

    def compare_monthly_bills(
        self,
        previous_period: BillPeriod,
        current_period: BillPeriod,
    ) -> dict:
        """Compare two bills and report changes without claiming their cause."""
        previous = previous_period.model_dump()
        current = current_period.model_dump()
        print(
            "TOOL CALL: compare_monthly_bills "
            f"({previous['month']} -> {current['month']})"
        )

        def change(current_value, previous_value):
            amount = current_value - previous_value
            percent = None if previous_value == 0 else amount / previous_value * 100
            return round(amount, 2), None if percent is None else round(percent, 1)

        usage_change, usage_percent = change(current["kwh"], previous["kwh"])
        bill_change, bill_percent = change(current["bill_amount"], previous["bill_amount"])

        if previous["billing_days"] is None or current["billing_days"] is None:
            previous_kwh_per_day = current_kwh_per_day = daily_usage_change = None
            previous_bill_per_day = current_bill_per_day = daily_bill_change = None
        else:
            previous_kwh_per_day = previous["kwh"] / previous["billing_days"]
            current_kwh_per_day = current["kwh"] / current["billing_days"]
            previous_bill_per_day = previous["bill_amount"] / previous["billing_days"]
            current_bill_per_day = current["bill_amount"] / current["billing_days"]
            daily_usage_change = current_kwh_per_day - previous_kwh_per_day
            daily_bill_change = current_bill_per_day - previous_bill_per_day
            previous_kwh_per_day = round(previous_kwh_per_day, 2)
            current_kwh_per_day = round(current_kwh_per_day, 2)
            previous_bill_per_day = round(previous_bill_per_day, 2)
            current_bill_per_day = round(current_bill_per_day, 2)
            daily_usage_change = round(daily_usage_change, 2)
            daily_bill_change = round(daily_bill_change, 2)

        result = {
            "previous_month": previous["month"],
            "current_month": current["month"],
            "usage_change_kwh": usage_change,
            "usage_change_percent": usage_percent,
            "bill_change_amount": bill_change,
            "bill_change_percent": bill_percent,
            "previous_kwh_per_day": previous_kwh_per_day,
            "current_kwh_per_day": current_kwh_per_day,
            "daily_usage_change_kwh": daily_usage_change,
            "previous_bill_per_day": previous_bill_per_day,
            "current_bill_per_day": current_bill_per_day,
            "daily_bill_change": daily_bill_change,
            "note": (
                "This comparison shows what changed; it does not identify the cause. "
                "Percent changes are null when the previous value was zero. Daily "
                "averages are null when either billing period length is unknown."
            ),
        }
        print("TOOL RESULT: compare_monthly_bills completed")
        return result
