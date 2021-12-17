import math
import queue
import random
import time
import threading
from typing import Iterator, Optional, Tuple

from rich import box
from rich.align import Align
from rich.layout import Layout
from rich.console import Console
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.columns import Columns
from river.compose import TransformerUnion
from river.feature_extraction import Agg


Stream = Iterator[Tuple[int, dict]]


def silly_stream() -> Stream:
    while True:
        yield {
            "t": (t := time.time()),
            "c": (c := random.choice(["a", "b"])),
            "d": (d := random.choice(["c", "d"])),
            "x": {"a": -10, "b": 10}[c] * (1 + math.cos(t)),
            "y": {"a": -10, "b": 10}[c] * (1 + math.sin(t)),
        }
        time.sleep(0.1)


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
        table.add_column(by, justify="center", no_wrap=True)
    table.add_column(agg.on)

    # The list() is here to copy the values of agg.groups, because the
    # size might change during iteration over items, due to the data
    # being updated in the background.
    for by, stat in list(agg.groups.items()):
        table.add_row(*by, f"{stat.get():,.5f}")

    table.box = box.SIMPLE_HEAD

    return table


class ETL(threading.Thread):
    def __init__(self, *aggs, stream: Stream):
        super().__init__()
        self.running = True
        self.stream = stream
        self.agg = TransformerUnion(*aggs)
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
        while self.running:
            for record in self.stream:
                self.agg.learn_one(record)
                self.n += 1


class Display(threading.Thread):
    def __init__(self, etl: ETL):
        super().__init__()
        self.running = True
        self.etl = etl

    def stop(self):
        self.running = False

    def run(self):
        def make_tables():
            return [
                _river_agg_to_rich_table(agg)
                for agg in self.etl.agg.transformers.values()
            ]

        with Live(refresh_per_second=10, screen=True, transient=True) as live:
            while self.running:

                layout = Layout()
                layout.split_column(
                    Layout(name="upper", ratio=99),
                    Layout(name="lower", ratio=1),
                )

                if percent_processed := self.etl._percent_processed is not None:
                    progress = ProgressBar(
                        total=self.etl.n + len(self.etl.stream), completed=self.etl.n
                    )
                    layout["lower"].update(progress)

                layout["upper"].update(
                    Align(Columns(make_tables()), align="center", vertical="middle")
                )
                # layout = Align(layout, vertical="middle")
                live.update(layout)


from river import stats

buffer = Buffer(silly_stream())
buffer.start()
time.sleep(5)

etl = ETL(
    Agg(on="x", by="c", how=stats.Mean()),
    Agg(on="x", by="d", how=stats.Mean()),
    stream=buffer,
)
etl.start()
time.sleep(3)
etl.stop()

display = Display(etl)
display.start()

time.sleep(10)

display.stop()
etl.stop()
buffer.stop()
