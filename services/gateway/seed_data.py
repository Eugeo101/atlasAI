"""
Seeds the units table with synthetic real-estate listings, and drops a
handful of plain-text "documents" on disk to stand in for brochures/payment
plans (real chunking + ingestion is Phase 2 — today these are just files
you can point a script at later).
"""
import random
from pathlib import Path
import pandas as pd

from faker import Faker
from models import Conversation, Unit, get_session

# fake = Faker()

CITIES = ["Cairo", "Riyadh", "Jeddah", "Dubai", "Alexandria", "Doha"]
DISTRICTS = ["Downtown", "New Capital", "Al Olaya", "Marina", "Business Bay", "Zamalek"]
UNIT_TYPES = ["apartment", "villa", "studio", "office", "townhouse"]

NUM_UNITS = 1000  # adjust between 500-2000 as your doc suggests
NUM_CONVERSATIONS = 20  # number of separate chat sessions to fake

# Fake user asks paired with a fake assistant reply — not meant to be
# realistic agent behavior (no tools/RAG exist yet), just enough shape
# to prove the conversations table round-trips real rows correctly.
SAMPLE_EXCHANGES = [
    ("Do you have any 2-bedroom apartments in Riyadh under 800k?",
     "Yes, I found a few 2-bedroom apartments in Al Olaya within that range. Want me to list them?"),
    ("What's the price difference between a studio and a 1-bedroom in Cairo?",
     "Studios in Cairo average lower than 1-bedrooms by roughly 20-30% depending on the district."),
    ("Is there a villa available in Dubai Marina?",
     "There's one villa listed in Marina currently marked as available."),
    ("Can you compare payment plans for two units?",
     "I can walk through the payment plan documents once they're linked — that part isn't wired up yet."),
    ("What's the average area for a townhouse?",
     "Townhouses in the current listings average around 220 sqm."),
]


def generate_unit():
    unit_type = random.choice(UNIT_TYPES)
    bedrooms = 0 if unit_type == "studio" else random.randint(1, 6)
    area = round(random.uniform(35, 450), 1)
    price = round(area * random.uniform(800, 4500), 2)

    return Unit(
        city=random.choice(CITIES),
        district=random.choice(DISTRICTS),
        unit_type=unit_type,
        price=price,
        bedrooms=bedrooms,
        bathrooms=max(1, bedrooms - random.randint(0, 1)),
        area_sqm=area,
        is_available=random.random() > 0.15,
        description=fake.paragraph(nb_sentences=4),
    )

def read_units(path):
    df = pd.read_csv(path)
    units = [
    Unit(
        city=row["city"],
        district=row["district"],
        unit_type=row["unit_type"],
        price=float(row["price"]),
        bedrooms=int(row["bedrooms"]),
        bathrooms=int(row["bathrooms"]),
        area_sqm=float(row["area_sqm"]),
        # Convert numeric 1.0/0.0/NaN to boolean or None
        is_available=(
            bool(row["is_available"]) if pd.notna(row["is_available"]) else None
        ),
        description=row["description"],
    )
    for _, row in df.iterrows()
    ]
    return units

def seed_units(session, n=NUM_UNITS, read_data=True):
    if read_data:
        units = read_units(path='eval//sample_units//units_data.csv')
    else:
        units = [generate_unit() for _ in range(n)]
    session.bulk_save_objects(units)
    session.commit()
    print(f"Seeded {n} units.")


def seed_conversations(session, n_sessions=NUM_CONVERSATIONS):
    """
    Fakes a handful of short chat sessions (2-4 turns each) so the
    conversations table has real rows to browse and query against.
    NOT meant to simulate real agent behavior — that's Phase 2's job
    once the actual agent loop exists.
    """
    total_rows = 0
    for i in range(n_sessions):
        session_id = f"session_{fake.uuid4()[:8]}"
        num_turns = random.randint(2, 4)
        for _ in range(num_turns):
            user_msg, assistant_msg = random.choice(SAMPLE_EXCHANGES)
            session.add(Conversation(session_id=session_id, role="user", content=user_msg))
            session.add(Conversation(session_id=session_id, role="assistant", content=assistant_msg))
            total_rows += 2
    session.commit()
    print(f"Seeded {total_rows} conversation rows across {n_sessions} sessions.")


def seed_sample_documents(out_dir="eval/sample_documents", n=30):
    """
    Plain-text stand-ins for brochures/payment plans. Phase 2 will replace
    these with real chunking + embedding into Qdrant — for now they just
    need to exist so the pipeline has something to point at later.
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        doc_type = random.choice(["brochure", "payment_plan", "faq"])
        content = f"{doc_type.upper()} #{i}\n\n" + fake.paragraph(nb_sentences=8)
        (path / f"{doc_type}_{i}.txt").write_text(content)
    print(f"Wrote {n} sample documents to {out_dir}/")


if __name__ == "__main__":
    session = get_session()
    seed_units(session, n=NUM_UNITS, read_data=True)
    # seed_conversations(session)
    # seed_sample_documents()
    session.close()