import json
from pathlib import Path
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Add the project root to sys.path to allow importing src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.cvat_parser import parse_cvat_xml

def main():
    xml_path = project_root / "data" / "annotations.xml"
    output_path = project_root / "data" / "processed" / "landmarks_clean.json"
    
    log.info("Parsing CVAT XML: %s", xml_path)
    
    try:
        records = parse_cvat_xml(xml_path)
    except FileNotFoundError as e:
        log.error("Error: %s", e)
        return
        
    log.info("Successfully parsed %d image records.", len(records))
    
    # Count some stats
    with_calibration = sum(1 for r in records if r.get("has_calibration"))
    with_landmarks = sum(1 for r in records if r.get("has_landmarks"))
    with_polygons = sum(1 for r in records if r.get("polygons"))
    
    log.info("Stats: %d have calibration, %d have landmarks, %d have polygons", 
             with_calibration, with_landmarks, with_polygons)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    log.info("Saving cleanly parsed records to %s", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"images": records}, f, indent=2)
        
    log.info("Done!")

if __name__ == "__main__":
    main()
