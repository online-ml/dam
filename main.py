import math
import queue
import random
import time
import threading
from typing import Iterator, Optional, Tuple

from rich.console import Console
from rich.table import Table
from river.feature_extraction import Agg


Stream = Iterator[Tuple[int, dict]]


def silly_stream() -> Stream:
    while True:
        yield {
            "t": (t := time.time()),
            "cat": (cat := random.choice(["a", "b"])),
            "x": {"a": -10, "b": 10}[cat] * (1 + math.cos(t)),
            "y": {"a": -10, "b": 10}[cat] * (1 + math.sin(t)),
        }
        time.sleep(1)


class Buffer(threading.Thread):
    def __init__(self, stream: Stream):
        super().__init__()
        self.running = True
        self.stream = stream
        self.records = queue.Queue()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            self.records.put(next(self.stream))

    def __iter__(self):
        while not self.records.empty():
            yield self.records.get()

    def __len__(self):
        return self.records.qsize()


def _river_agg_to_rich_table(agg: Agg) -> Table:
    table = Table(title=agg.feature_name)

    for by in agg.by:
        table.add_column("Released", justify="center", no_wrap=True)
    table.add_column("")

    for by, stat in agg.groups.items():
        table.add_row(*by, f"{stat.get():,.5f}")

    return table


class ETL(threading.Thread):
    def __init__(self, agg: Agg, stream: Stream):
        super().__init__()
        self.running = True
        self.stream = stream
        self.agg = agg
        self.n = 0

    def stop(self):
        self.running = False

    @property
    def _percent_processed(self) -> Optional[float]:
        """Return the % of records that have been processed.

        This only makes sense if the stream has some notion of state. It only works
        if the stream can tell us how much data it is holding for us to process. Raw
        streams do no do this. A buffer needs to be wrapped around the stream for this
        to be possible.

        """
        try:
            return self.n / (self.n + len(self.stream))
        except AttributeError:
            return None

    def run(self):
        console = Console()

        with console.screen(style="bold white on black") as screen:
            while self.running:
                for record in self.stream:
                    self.agg.learn_one(record)
                    screen.update(_river_agg_to_rich_table(self.agg))

            # for record in self.stream:
            #     self.agg.learn_one(record)
            #     self.n += 1
            #     msg = ""
            #     if (_percent_processed := self._percent_processed) :
            #         msg += f"{_percent_processed:.2%} processed\n"
            #     msg += str(self.agg.transform_one(record))
            #     print(msg, end="\r")


from river import stats

buffer = Buffer(silly_stream())
buffer.start()

etl = ETL(agg=Agg(on="x", by="cat", how=stats.Mean()), stream=buffer)
etl.start()

time.sleep(10)
etl.stop()

buffer.stop()
