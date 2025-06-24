import re

def detect_product_type(name):
    name = name.lower()

    if re.search(r"3[-\s]?way.*ball valve|ball valve.*3[-\s]?way", name):
        return "3-Way Ball Valve"
    if re.search(r"2[-\s]?way.*ball valve|ball valve.*2[-\s]?way", name):
        return "2-Way Ball Valve"
    if re.search(r"\bball valve\b", name) and not re.search(r"3[-\s]?way.*ball valve|ball valve.*3[-\s]?way|2[-\s]?way.*ball valve|ball valve.*2[-\s]?way", name):
        return "Ball Valve"

    match = re.search(
        r"(check valve|gasket|manifold valve|butterfly valve|steam trap|pressure reducing valve|solenoid valve|actuator|limit switch|positioner|transducer|regulator|control valve|gate valve|need valve|needle valve|pneumatic valve|backflow preventor|backflow preventer|air lubricator|spacer|bracket|air filter|angle seat valve|diaphragm valve|plug valve|globe valve|block and bleed valve|safety relief valve|fitting|hanger|gauge|flow meter|transmitter|level measurement|mag meter|sight flow indicator|sight glass|vacuum transmitter|temperature sensor|pressure relief valve|sample valve|sight glasses|gauges|pressure transmitter|fittings|sanitary clamp gasket|manual air foot valve|air cylinder reed switche|air cylinder|filter|strainer|digital timer|4 way valve|lockout valve|lubricator|level radar|flow control|degassing valve|vacuum breaker|muffler|metering valve|anti-siphon valve|air release valve|manifold|adapter|clamp|ferrule|wing nut|sleeve|end cap|manual override|position indicator|elbow|breather vent)",
        name
    )
    if match:
        return match.group(1).title()

    if re.search(r"foot valve|swing check", name):
        return "Check Valve"
    if re.search(r"\bsolenoid\b", name):
        return "Solenoid Valve"
    if re.search(r"\bcylinder\b", name):
        return "Air Cylinder"
    if re.search(r"manual air valve", name):
        return "Pneumatic Valve"
    if re.search(r"din coil|coil conduit", name):
        return "Solenoid Coil"
    if re.search(r"explosion proof coil|explosion proof.*coil", name):
        return "Solenoid Coil"
    if re.search(r"union valve", name):
        return "Air Valve"
    if re.search(r"[0-9]+ ?psi|glycerin filled|back mount", name):
        return "Gauge"
    if re.search(r"sanitary relief valve", name):
        return "Pressure Relief Valve"
    if re.search(r"\badapter\b", name):
        return "Adapter"
    if re.search(r"\bclamp\b", name):
        return "Clamp"
    if re.search(r"\bferrule\b", name):
        return "Ferrule"
    if re.search(r"\bwing nut\b", name):
        return "Wing Nut"
    if re.search(r"\bgasket\b", name):
        return "Gasket"
    if re.search(r"\bsleeve\b", name):
        return "Sleeve"
    if re.search(r"\bend cap\b", name):
        return "End Cap"
    if re.search(r"manual override|declutchable", name):
        return "Manual Override"
    if re.search(r"position indicator", name):
        return "Position Indicator"
    if re.search(r"\belbow\b", name):
        return "Elbow"
    if re.search(r"\bbreather vent\b", name):
        return "Breather Vent"
    if re.search(r"\bair valve[s]?\b", name):
        return "Air Valve"
    if re.search(r"(slip[-\s]?on|weld[-\s]?neck|blind|lap[-\s]?joint|threaded)? ?flange[s]?", name):
        return "Flange"

    return "Component"