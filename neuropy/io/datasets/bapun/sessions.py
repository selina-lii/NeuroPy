"""Which session directories this dataset holds, by study arm."""
from __future__ import annotations

import os
from pathlib import Path

BASEDIR = Path(os.path.expanduser("~/Documents/ms_synchrony/bapun"))

NSD = ['RatJ/Day2', 'RatK/Day2', 'RatN/Day2', 'RatS/Day2NSD', 'RatR/Day1NSD',
       'RatU/RatUDay2NSD', 'RatV/RatVDay1NSD', 'RatV/RatVDay3NSD']

SD = ['RatJ/Day1', 'RatK/Day1', 'RatN/Day1', 'RatS/Day3SD', 'RatR/Day2SD',
      'RatU/RatUDay1SD', 'RatU/RatUDay4SD', 'RatV/RatVDay2SD']

ARMS = {'NSD': NSD, 'SD': SD}
