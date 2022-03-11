import math

import pandas as pd


def timedelta_to_str(td: pd.Timedelta) -> str:
    """Construct a tight string representation for the given timedelta arg.

    Parameters
    ----------
    td: pd.Timedelta
        The timedelta for which the string representation is constructed

    Returns
    -------
    str:
        The tight string bounds of format '$d-$h$m$s.$ms'.

    """
    out_str = ""

    # Edge case if we deal with negative
    if td < pd.Timedelta(seconds=0):
        td *= -1
        out_str += "NEG"

    # Note: this must happen after the *= -1
    c = td.components
    if c.days > 0:
        out_str += f"{c.days}D"
    if c.hours > 0 or c.minutes > 0 or c.seconds > 0 or c.milliseconds > 0:
        out_str += "_" if len(out_str) else ""

    if c.hours > 0:
        out_str += f"{c.hours}h"
    if c.minutes > 0:
        out_str += f"{c.minutes}m"
    if c.seconds > 0:
        if c.milliseconds:
            out_str += (
                f"{c.seconds}.{str(c.milliseconds / 1000).split('.')[-1].rstrip('0')}s"
            )
        else:
            out_str += f"{c.seconds}s"
    elif c.milliseconds > 0:
        out_str += f"{str(c.milliseconds)}ms"
    if c.microseconds > 0:
        out_str += f"{str(c.microseconds)}us"
    if c.nanoseconds > 0:
        out_str += f"{str(c.nanoseconds)}ns"
    return out_str


def round_td_str(td: pd.Timedelta) -> str:
    for t_s in ["D", "H", "min", "s", "ms", "us", "ns"]:
        if td > 0.95 * pd.Timedelta(f"1{t_s}"):
            return timedelta_to_str(td.round(t_s))


def round_number_str(number: float) -> str:
    if number > 0.95:
        for unit, scaling in [("M", int(1e6)), ("k", int(1e3))]:
            if number / scaling > 0.95:
                return f"{round(number / scaling)}{unit}"
        return str(round(number))
    # we have a number < 1 --> round till nearest non-zero digit
    return str(round(number, 1 + abs(int(math.log10(number)))))
