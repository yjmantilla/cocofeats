import json
from cocofeats.loggers import get_logger

log = get_logger(__name__)

def save_dict_to_json(jsonfile,data):
    with open(jsonfile, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    log.debug("Saved JSON file", file=jsonfile)
