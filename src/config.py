"""Shared settings for the proof of concept."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SYNTH = DATA / "synthetic"      # raw mock exports, one folder per vertical
ERP = DATA / "mock_erp"         # SQLite mock ERP databases
RESULTS = DATA / "results"      # pipeline outputs the dashboard reads

WEEKS = 156                     # 3 years of weekly history
END_DATE = "2026-11-16"         # last week in the data (a Monday)
HORIZON = 13                    # backtest horizon in weeks (about a quarter)
N_WINDOWS = 3                   # backtest windows; the last 39 weeks are tested
REVIEW_WEEKS = 1                # weekly ordering cycle
SEED = 7

VERTICALS = {
    "ag_ope": {
        "label": "Ag and outdoor power equipment",
        "prefix": "AG", "n_items": 1200,
        "locations": [("KAN", "Main warehouse", "warehouse", None),
                      ("DSM", "Des Moines branch", "branch", None),
                      ("FAR", "Fargo branch", "branch", None)],
        "seasonal_share": 0.60, "cost": (8, 900), "lead_time": (2, 8), "import_share": 0.15,
        "categories": {"Planter": 16, "Combine": 37, "Tractor": 20, "Mower": 26, "Snow": 1},
        "makes": ["John Deere", "Case IH", "Kinze", "New Holland", "Kubota", "Toro"],
        "condition_used_share": 0.05,
    },
    "powersports": {
        "label": "Powersports parts",
        "prefix": "PS", "n_items": 2000,
        "locations": [("WH1", "Distribution center", "warehouse", None)],
        "seasonal_share": 0.70, "cost": (3, 450), "lead_time": (2, 6), "import_share": 0.30,
        "categories": {"ATV": 22, "UTV": 24, "Motorcycle": 20, "Snowmobile": 2},
        "makes": ["Polaris", "Can-Am", "Honda", "Yamaha", "Kawasaki", "Arctic Cat"],
        "condition_used_share": 0.0,
    },
    "industrial": {
        "label": "Industrial bearing and fluid power",
        "prefix": "IN", "n_items": 1000,
        "locations": [("DET", "Detroit", "branch", None), ("FLT", "Flint", "branch", None),
                      ("GRR", "Grand Rapids", "branch", None), ("TOL", "Toledo", "branch", None)],
        "seasonal_share": 0.05, "cost": (2, 1500), "lead_time": (3, 12), "import_share": 0.10,
        "categories": {"Bearings": 0, "Seals": 0, "Hydraulic valves": 0, "Belts and sheaves": 0, "Gearboxes": 0},
        "makes": ["Pump", "Gearbox", "Conveyor", "Press", "Compressor"],
        "condition_used_share": 0.0,
    },
    "auto_hd": {
        "label": "Auto and heavy-duty WD",
        "prefix": "HD", "n_items": 1200,
        "locations": [("DC", "Central warehouse", "warehouse", None), ("DBN", "Dearborn branch", "branch", "DC"),
                      ("LNP", "Lincoln Park branch", "branch", "DC"), ("WAR", "Warren branch", "branch", "DC")],
        "seasonal_share": 0.15, "cost": (3, 600), "lead_time": (1, 4), "import_share": 0.20,
        "categories": {"Brakes": 0, "Air system": 0, "Electrical": 50, "Suspension": 0, "Cooling": 28, "Lighting": 0},
        "makes": ["Freightliner", "Peterbilt", "Kenworth", "Volvo", "International", "Ford", "GM"],
        "condition_used_share": 0.0,
    },
}
# Category value = peak week of the year for seasonal parts (0 = not seasonal by nature).
