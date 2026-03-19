import pandas as pd

# Define the supply chain mapping based on Sri Lankan agricultural economics (HARTI/DECs)
# Upcountry vegetables: Carrot, Leeks, Cabbage, Beetroot, Green Beans
# Lowcountry vegetables: Pumpkin, Brinjals, Ladies Fingers, Snake Gourd, Ash Plantains, Tomatoes, Green Chillies

mappings = [
    # --- COLOMBO (Terminal Market: Manning Market / Peliyagoda) ---
    {'Market': 'Colombo', 'Vegetable': 'CARROT', 'Source': 'Nuwaraeliya (Primary), Badulla (Secondary)'},
    {'Market': 'Colombo', 'Vegetable': 'LEEKS', 'Source': 'Nuwaraeliya (Primary), Badulla (Secondary)'},
    {'Market': 'Colombo', 'Vegetable': 'CABBAGE', 'Source': 'Nuwaraeliya (Primary), Badulla (Secondary)'},
    {'Market': 'Colombo', 'Vegetable': 'BEETROOT', 'Source': 'Nuwaraeliya (Primary), Badulla (Secondary)'},
    {'Market': 'Colombo', 'Vegetable': 'GREEN BEANS', 'Source': 'Nuwaraeliya (Primary), Badulla (Secondary)'},
    
    {'Market': 'Colombo', 'Vegetable': 'PUMPKIN', 'Source': 'Dambulla (Primary), Anuradhapura (Secondary)'},
    {'Market': 'Colombo', 'Vegetable': 'BRINJALS', 'Source': 'Dambulla (Primary), Embilipitiya (Secondary)'},
    {'Market': 'Colombo', 'Vegetable': 'LADIES FINGERS', 'Source': 'Dambulla (Primary), Thambuththegama (Secondary)'},
    {'Market': 'Colombo', 'Vegetable': 'SNAKE GOURD', 'Source': 'Dambulla (Primary)'},
    {'Market': 'Colombo', 'Vegetable': 'ASH PLANTAINS', 'Source': 'Dambulla (Primary), Embilipitiya (Secondary)'},
    {'Market': 'Colombo', 'Vegetable': 'TOMATOES', 'Source': 'Dambulla (Primary)'},
    {'Market': 'Colombo', 'Vegetable': 'GREEN CHILLIES', 'Source': 'Dambulla (Primary), Puttalam (Secondary)'},

    # --- MEEGODA (Dedicated Economic Centre - Terminal) ---
    # Functions nearly identical to Colombo but handles suburban Southern/Eastern flanks
    {'Market': 'Meegoda', 'Vegetable': 'CARROT', 'Source': 'Nuwaraeliya'},
    {'Market': 'Meegoda', 'Vegetable': 'LEEKS', 'Source': 'Nuwaraeliya'},
    {'Market': 'Meegoda', 'Vegetable': 'CABBAGE', 'Source': 'Nuwaraeliya'},
    {'Market': 'Meegoda', 'Vegetable': 'BEETROOT', 'Source': 'Nuwaraeliya'},
    {'Market': 'Meegoda', 'Vegetable': 'GREEN BEANS', 'Source': 'Nuwaraeliya'},
    
    {'Market': 'Meegoda', 'Vegetable': 'PUMPKIN', 'Source': 'Dambulla'},
    {'Market': 'Meegoda', 'Vegetable': 'BRINJALS', 'Source': 'Dambulla, Embilipitiya'},
    {'Market': 'Meegoda', 'Vegetable': 'LADIES FINGERS', 'Source': 'Dambulla'},
    {'Market': 'Meegoda', 'Vegetable': 'SNAKE GOURD', 'Source': 'Dambulla'},
    {'Market': 'Meegoda', 'Vegetable': 'ASH PLANTAINS', 'Source': 'Dambulla, Embilipitiya'},
    {'Market': 'Meegoda', 'Vegetable': 'TOMATOES', 'Source': 'Dambulla'},
    {'Market': 'Meegoda', 'Vegetable': 'GREEN CHILLIES', 'Source': 'Dambulla'},

    # --- KALUTHARA (Coastal Retail) ---
    # Kaluthara gets most of its stock routed down from Colombo/Meegoda, tracing back to the same hubs.
    {'Market': 'Kaluthara', 'Vegetable': 'CARROT', 'Source': 'Nuwaraeliya (via Colombo)'},
    {'Market': 'Kaluthara', 'Vegetable': 'LEEKS', 'Source': 'Nuwaraeliya (via Colombo)'},
    {'Market': 'Kaluthara', 'Vegetable': 'CABBAGE', 'Source': 'Nuwaraeliya (via Colombo)'},
    {'Market': 'Kaluthara', 'Vegetable': 'BEETROOT', 'Source': 'Nuwaraeliya (via Colombo)'},
    {'Market': 'Kaluthara', 'Vegetable': 'GREEN BEANS', 'Source': 'Nuwaraeliya (via Colombo)'},
    
    {'Market': 'Kaluthara', 'Vegetable': 'PUMPKIN', 'Source': 'Dambulla (via Colombo), Embilipitiya'},
    {'Market': 'Kaluthara', 'Vegetable': 'BRINJALS', 'Source': 'Embilipitiya, Dambulla'},
    {'Market': 'Kaluthara', 'Vegetable': 'LADIES FINGERS', 'Source': 'Dambulla (via Colombo)'},
    {'Market': 'Kaluthara', 'Vegetable': 'SNAKE GOURD', 'Source': 'Dambulla (via Colombo)'},
    {'Market': 'Kaluthara', 'Vegetable': 'ASH PLANTAINS', 'Source': 'Embilipitiya'},
    {'Market': 'Kaluthara', 'Vegetable': 'TOMATOES', 'Source': 'Dambulla (via Colombo)'},
    {'Market': 'Kaluthara', 'Vegetable': 'GREEN CHILLIES', 'Source': 'Dambulla (via Colombo)'},

    # --- MATHARA (Southern Deep Coastal) ---
    {'Market': 'Mathara', 'Vegetable': 'CARROT', 'Source': 'Badulla (Primary via Southern hills), Nuwaraeliya'},
    {'Market': 'Mathara', 'Vegetable': 'LEEKS', 'Source': 'Badulla, Nuwaraeliya'},
    {'Market': 'Mathara', 'Vegetable': 'CABBAGE', 'Source': 'Badulla, Nuwaraeliya'},
    {'Market': 'Mathara', 'Vegetable': 'BEETROOT', 'Source': 'Badulla, Nuwaraeliya'},
    {'Market': 'Mathara', 'Vegetable': 'GREEN BEANS', 'Source': 'Badulla, Nuwaraeliya'},
    
    {'Market': 'Mathara', 'Vegetable': 'PUMPKIN', 'Source': 'Hambanthota, Embilipitiya'},
    {'Market': 'Mathara', 'Vegetable': 'BRINJALS', 'Source': 'Embilipitiya, Hambanthota'},
    {'Market': 'Mathara', 'Vegetable': 'LADIES FINGERS', 'Source': 'Embilipitiya, Hambanthota'},
    {'Market': 'Mathara', 'Vegetable': 'SNAKE GOURD', 'Source': 'Embilipitiya, Hambanthota'},
    {'Market': 'Mathara', 'Vegetable': 'ASH PLANTAINS', 'Source': 'Embilipitiya, Hambanthota'},
    {'Market': 'Mathara', 'Vegetable': 'TOMATOES', 'Source': 'Embilipitiya, Dambulla'},
    {'Market': 'Mathara', 'Vegetable': 'GREEN CHILLIES', 'Source': 'Embilipitiya, Hambanthota'},

    # --- KANDY (Central Highland Border) ---
    {'Market': 'Kandy', 'Vegetable': 'CARROT', 'Source': 'Nuwaraeliya (Direct)'},
    {'Market': 'Kandy', 'Vegetable': 'LEEKS', 'Source': 'Nuwaraeliya (Direct)'},
    {'Market': 'Kandy', 'Vegetable': 'CABBAGE', 'Source': 'Nuwaraeliya (Direct)'},
    {'Market': 'Kandy', 'Vegetable': 'BEETROOT', 'Source': 'Nuwaraeliya (Direct)'},
    {'Market': 'Kandy', 'Vegetable': 'GREEN BEANS', 'Source': 'Nuwaraeliya (Direct)'},
    
    {'Market': 'Kandy', 'Vegetable': 'PUMPKIN', 'Source': 'Dambulla (Direct)'},
    {'Market': 'Kandy', 'Vegetable': 'BRINJALS', 'Source': 'Dambulla (Direct)'},
    {'Market': 'Kandy', 'Vegetable': 'LADIES FINGERS', 'Source': 'Dambulla (Direct)'},
    {'Market': 'Kandy', 'Vegetable': 'SNAKE GOURD', 'Source': 'Dambulla (Direct)'},
    {'Market': 'Kandy', 'Vegetable': 'ASH PLANTAINS', 'Source': 'Dambulla (Direct)'},
    {'Market': 'Kandy', 'Vegetable': 'TOMATOES', 'Source': 'Dambulla (Direct)'},
    {'Market': 'Kandy', 'Vegetable': 'GREEN CHILLIES', 'Source': 'Dambulla (Direct)'},

    # --- KURUNEGALA (North Western Node) ---
    {'Market': 'Kurunegala', 'Vegetable': 'CARROT', 'Source': 'Dambulla (Transit), Nuwaraeliya'},
    {'Market': 'Kurunegala', 'Vegetable': 'LEEKS', 'Source': 'Dambulla (Transit), Nuwaraeliya'},
    {'Market': 'Kurunegala', 'Vegetable': 'CABBAGE', 'Source': 'Dambulla (Transit), Nuwaraeliya'},
    {'Market': 'Kurunegala', 'Vegetable': 'BEETROOT', 'Source': 'Dambulla (Transit), Nuwaraeliya'},
    {'Market': 'Kurunegala', 'Vegetable': 'GREEN BEANS', 'Source': 'Dambulla (Transit), Nuwaraeliya'},
    
    {'Market': 'Kurunegala', 'Vegetable': 'PUMPKIN', 'Source': 'Dambulla, Thambuththegama'},
    {'Market': 'Kurunegala', 'Vegetable': 'BRINJALS', 'Source': 'Dambulla, Local production'},
    {'Market': 'Kurunegala', 'Vegetable': 'LADIES FINGERS', 'Source': 'Dambulla, Local production'},
    {'Market': 'Kurunegala', 'Vegetable': 'SNAKE GOURD', 'Source': 'Dambulla, Local production'},
    {'Market': 'Kurunegala', 'Vegetable': 'ASH PLANTAINS', 'Source': 'Dambulla'},
    {'Market': 'Kurunegala', 'Vegetable': 'TOMATOES', 'Source': 'Dambulla'},
    {'Market': 'Kurunegala', 'Vegetable': 'GREEN CHILLIES', 'Source': 'Dambulla, Puttalam'}
]

# Convert to DataFrame to pretty print
df_map = pd.DataFrame(mappings)

# Write to a clean markdown file for user reference
with open('Supply_Chain_Mapping.md', 'w') as f:
    f.write("# Sri Lanka Vegetable Supply Chain Mapping (HARTI/DEC Based)\n\n")
    current_market = ""
    for index, row in df_map.iterrows():
        if row['Market'] != current_market:
            current_market = row['Market']
            f.write(f"\n### {current_market} Market\n")
            f.write("| Vegetable | Primary Origin / Source (Producer Market) |\n")
            f.write("| :--- | :--- |\n")
        
        f.write(f"| {row['Vegetable']} | {row['Source']} |\n")

print("Created markdown output file.")
