"""
MIT License
Copyright (c) 2018 Free TNT
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import logging
import time
from typing import Optional, Callable, Any


def _current_nanoseconds() -> int:
    nanoseconds: int
    try:
        nanoseconds = time.perf_counter_ns()
    except AttributeError:
        nanoseconds = int(time.perf_counter() * 1_000_000_000)
    return nanoseconds


# taken from https://github.com/ravener/stopwatch.py/blob/master/stopwatch/stopwatch.py

# Ported from https://github.com/dirigeants/klasa/blob/master/src/lib/util/Stopwatch.js
class Stopwatch:
    def __init__(self) -> None:
        self._start: int = _current_nanoseconds()
        self._end: Optional[int] = None

    def timeit(self, callback: Callable[[], Any]) -> str:
        """
        :param callback: function to execute
        :return: string describing time taken to execute `callback()`
        """
        self.stop()
        self.start()
        callback()
        self.stop()
        return str(self)

    @property
    def duration_ns(self) -> int:
        # not supported in Python 3.6
        return self._end - self._start if self._end else _current_nanoseconds() - self._start

    @property
    def running(self) -> bool:
        return self._end is None

    def restart(self) -> 'Stopwatch':
        self._start = _current_nanoseconds()
        self._end = None
        return self

    def reset(self) -> 'Stopwatch':
        self._start = _current_nanoseconds()
        self._end = self._start
        return self

    def start(self) -> 'Stopwatch':
        if not self.running:
            self._start = _current_nanoseconds() - self.duration_ns
            self._end = None
        return self

    def stop(self) -> 'Stopwatch':
        if self.running:
            self._end = _current_nanoseconds()
        return self

    def nanoseconds(self) -> int:
        return self.duration_ns

    def microseconds(self) -> float:
        return self.duration_ns / 1_000.0

    def milliseconds(self) -> float:
        return self.duration_ns / 1_000_000.0

    def seconds(self) -> float:
        return self.duration_ns / 1_000_000_000.0

    def nanoseconds_str(self) -> str:
        return f'{self.nanoseconds()}'

    def microseconds_str(self, precision: int = 2) -> str:
        return f'{self.microseconds():.{precision}f}'

    def milliseconds_str(self, precision: int = 2, width: Optional[int] = None) -> str:
        width_spec = '' if width == None else width
        return f'{self.milliseconds():{width_spec}.{precision}f}'

    def seconds_str(self, precision: int = 2) -> str:
        return f'{self.seconds():.{precision}f}'

    def __str__(self) -> str:
        time_ = self.milliseconds()
        if time_ >= 1000:
            return "{:.2f} s".format(time_ / 1000)
        if time_ >= 1:
            return "{:.2f} ms".format(time_)
        if time_ >= 0.001:
            return "{:.2f} μs".format(time_ * 1000)
        return "{:.2f} ns".format(time_ * 1_000_000)

    def time_in_units(self,
                      # units: Literal['s', 'ms', 'us', 'ns']) -> str: # Literal not supported in Python 3.7
                      units: str) -> str:
        ns = self.nanoseconds()
        if units == 's':
            return f'{ns / 10 ** 9:.2f} s'
        elif units == 'ms':
            return f'{ns / 10 ** 6:.2f} ms'
        elif units == 'us':
            return f'{ns / 10 ** 3:.2f} μs'
        elif units == 'ns':
            return f'{ns:.2f} ns'
        else:
            raise ValueError(f"units = {units} is not a legal unit, please choose one of "
                             f"'s', 'ms', 'us', 'ns'")

    def log(self, msg: str,
            # units: Optional[Literal['s', 'ms', 'us', 'ns']] = None, # Literal not supported in Python 3.7
            units: Optional[str] = None,
            restart: bool = True,
            logger: Optional[logging.Logger] = None,
            ) -> None:
        """
        Useful for timing statements/blocks of statements via the following:

        .. code-block:: python

            sw = Stopwatch()
            # statement(s) to time
            sw.log('my statements')

        Which will print something like this to the screen:

          time for my statements: 47.80μs

        :param msg:
            message to indicate what is being timed ("my statements" in the example above)
        :param units:
            units ('s', 'ms', 'us', 'ns') to use. If not specified then a reasonable default is chosen.
        :param restart:
            whether to restart this Stopwatch instance after logging
            (useful for timing several statements in a row)
        :param logger:
            if specified, writes log message to `logger` instead of printing to stdout
        """
        if logger is None:
            print_local = print
        else:
            def print_local(msg: str) -> None:
                logger.info(msg)

        if units is None:
            print_local(f'time for {msg}: {self}')
        else:
            print_local(f'time for {msg}: {self.time_in_units(units)}')

        if restart:
            self.restart()
