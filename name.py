import re
from type import detect_product_type

def apollo(readable_name, pdf_text):
    valve_naming_types = [
        "2-Way Ball Valve", "3-Way Ball Valve", "Backflow Preventer",
        "Ball Valve", "Check Valve", "Safety Relief Valve"
    ]

    product_type = detect_product_type(readable_name)
    if product_type not in valve_naming_types:
        return f"{readable_name.strip()} – {product_type}"

    # Extract Size
    size_matches = re.findall(r'\d+(?:-\d+)?(?:/\d+)?\s*"', readable_name)
    size = max(size_matches, key=len).strip() if size_matches else ""


    # Brand + Series
    brand_series_match = re.search(r'([A-Z][a-zA-Z]+\s+[0-9A-Za-z\-]+)', readable_name)
    brand_series = brand_series_match.group(1).strip() if brand_series_match else readable_name.strip()

    # Material
    material = ""
    material_map = [
        ("lead[-\\s]?free.*brass", "Lead-Free Brass"),
        ("brass", "Brass"),
        ("stainless steel", "Stainless Steel"),
        ("carbon steel", "Carbon Steel"),
        ("ductile iron", "Ductile Iron"),
        ("cast iron", "Cast Iron"),
    ]
    for pattern, label in material_map:
        if re.search(pattern, readable_name, re.IGNORECASE):
            material = label
            break

    # Connection Type
    connection_map = {
        r'npt|fnpt|threaded': "Threaded",
        r'flanged': "Flanged",
        r'socket\s*weld': "Socket Weld",
        r'butt\s*weld': "Butt Weld",
        r'solder\s*end': "Solder End",
        r'press\s*end': "Press End"
    }
    connection = next((label for pattern, label in connection_map.items() if re.search(pattern, readable_name, re.IGNORECASE)), "")

    # Port Type
    port = ""
    port_match = re.search(r'(Full[-\s]?Port|Standard[-\s]?Port|Reduced[-\s]?Port)', readable_name, re.IGNORECASE)
    if port_match:
        port = port_match.group(1).replace("-", " ").title()

    # Design Features (like 2-piece, spring-loaded, etc.)
    design_features = []
    if re.search(r'\b2[-\s]*piece\b', readable_name, re.IGNORECASE):
        design_features.append("2-Piece")
    elif re.search(r'\b3[-\s]*piece\b', readable_name, re.IGNORECASE):
        design_features.append("3-Piece")
    if re.search(r'spring[-\s]?loaded', readable_name, re.IGNORECASE):
        design_features.append("Spring-Loaded")
    if re.search(r'in[-\s]?line', readable_name, re.IGNORECASE):
        design_features.append("In-Line")
    if re.search(r'direct mount', readable_name, re.IGNORECASE):
        design_features.append("Direct Mount")
    if re.search(r'pneumatic', readable_name, re.IGNORECASE):
        design_features.append("Pneumatic")

    features = " ".join(design_features).strip()

    # "with ___" feature extraction (at the end of name)
    with_feature_match = re.findall(r'\bwith\s+[a-zA-Z0-9\s\-]+', readable_name, re.IGNORECASE)
    with_feature_text = " ".join(w.strip().title() for w in with_feature_match)

    # Compose final name
    info_parts = [material, connection, port, features, product_type]
    info_main = " ".join(part for part in info_parts if part)

    prefix = f'{size} ' if size else ''
    if with_feature_text:
        return f"{prefix}{brand_series} – {info_main} {with_feature_text}".strip()
    else:
        return f"{prefix}{brand_series} – {info_main}".strip()
