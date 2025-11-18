from Task1 import predict_price
import pandas as pd

def price_storage_contract(
        injection_date,
        withdrawal_date,
        injection_volume,
        withdrawal_volume,
        max_volume,
        injection_rate,
        withdrawal_rate,
        storage_cost_per_unit_per_day,
        injection_cost_per_unit=0.0,
        withdrawal_cost_per_unit=0.0,
        injection_price_override=None,
        withdrawal_price_override=None
):
    inventory = 0
    total_purchase = total_revenue = 0
    total_storage = total_injection_cost = total_withdrawal_cost = 0

    inject_date = pd.to_datetime(injection_date)
    withdraw_date = pd.to_datetime(withdrawal_date)
    events = [
        (inject_date, min(injection_volume, injection_rate), "inject", injection_price_override),
        (withdraw_date, -min(withdrawal_volume, withdrawal_rate), "withdraw", withdrawal_price_override)
    ]
    events.sort()
    last_date = events[0][0]

    for date, volume_change, action, override_price in events:
        days = (date - last_date).days
        total_storage += min(inventory, max_volume) * storage_cost_per_unit_per_day * days
        last_date = date

        price = override_price if override_price is not None else predict_price(date)
        volume = abs(volume_change)

        if action == "inject":
            inventory += volume
            inventory = min(inventory, max_volume)
            total_purchase += price * volume
            total_injection_cost += injection_cost_per_unit * volume
        else:
            volume = min(volume, inventory)
            inventory -= volume
            total_revenue += price * volume
            total_withdrawal_cost += withdrawal_cost_per_unit * volume

    return round(total_revenue - total_purchase - total_storage - total_injection_cost - total_withdrawal_cost, 2)


# TEST CASE
if __name__ == "__main__":
    result = price_storage_contract(
        injection_date="2024-07-01",
        withdrawal_date="2024-11-01",
        injection_volume=4_000_000,
        withdrawal_volume=4_000_000,
        max_volume=4_000_000,
        injection_rate=4_000_000,
        withdrawal_rate=4_000_000,
        storage_cost_per_unit_per_day=200000 / (4_000_000 * 122),
        injection_cost_per_unit=0.0,
        withdrawal_cost_per_unit=0.0,
        injection_price_override=12.349200136506317,
        withdrawal_price_override=13.096962762701084
    )

    print(f"Estimated Contract Value: {result:.2f}")
