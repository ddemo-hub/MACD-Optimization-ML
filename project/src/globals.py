from dataclasses import dataclass
import pathlib

@dataclass
class Globals():
    project_path = pathlib.Path(__file__).parent.parent
    
    klines_path = f"{project_path}/klines"