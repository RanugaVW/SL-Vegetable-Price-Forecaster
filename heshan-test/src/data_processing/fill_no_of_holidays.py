import argparse
import csv
import datetime as dt
import holidays

def first_friday(year: int) -> dt.date:
    d = dt.date(year, 1, 1)
    while d.weekday() != 4:  # Friday
        d += dt.timedelta(days=1)
    return d

def count_week_holidays(year: int, week_num: int, lk_holidays) -> int:
    start = first_friday(year) + dt.timedelta(days=7 * (week_num - 1))
    return sum(1 for i in range(7) if (start + dt.timedelta(days=i)) in lk_holidays)

def parse_week(week_value: str) -> int:
    return int(str(week_value).strip().lower().replace("w", ""))

def main() -> None:
    p = argparse.ArgumentParser(description="Fill no_of_holidays in CSV using python-holidays.")
    p.add_argument("--input", default='data/main.csv', help="Input CSV path")
    p.add_argument("--output", default='data/main_filled.csv', help="Output CSV path")
    args = p.parse_args()

    with open(args.input, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("Input CSV has no rows.")

    keys = {(int(r["year"]), parse_week(r["week"])) for r in rows}
    years = sorted({y for y, _ in keys})
    
    lk_holidays = holidays.country_holidays("LK", years=range(min(years), max(years) + 2))

    holiday_count_by_key = {
        (y, w): count_week_holidays(y, w, lk_holidays)
        for y, w in keys
    }

    for r in rows:
        y = int(r["year"])
        w = parse_week(r["week"])
        r["no_of_holidays"] = str(holiday_count_by_key[(y, w)])

    fieldnames = list(rows[0].keys())
    if "no_of_holidays" not in fieldnames:
        insert_at = 7 if len(fieldnames) >= 7 else len(fieldnames)
        fieldnames.insert(insert_at, "no_of_holidays")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")

if __name__ == "__main__":
    main()
