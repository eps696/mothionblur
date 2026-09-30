"""Terminal progress bar with ETA, used for Atlas job submission and video encoding."""
import sys
import time
from shutil import get_terminal_size


def hms(sec):
    """Format seconds as h:mm:ss"""
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return '%d:%02d:%02d' % (h, m, s)


class progbar:
    def __init__(self, total, width=40):
        self.total = max(1, total)          # avoid /0 if called with total=0
        self.width = min(width, get_terminal_size().columns - 30)
        self.done = 0
        self.start_time = time.time()

    def upd(self, uprows=0):
        self.done += 1
        elapsed = time.time() - self.start_time
        frac = self.done / self.total
        eta = elapsed * (1 - frac) / frac if frac > 0 else 0
        end_time = time.strftime('%H:%M:%S', time.localtime(self.start_time + elapsed + eta))
        rate = elapsed / self.done
        filled = int(self.width * frac)
        bar = '#' * filled + '-' * (self.width - filled)
        if uprows: sys.stdout.write('\033[%dA' % uprows)  # move up over extra lines
        sys.stdout.write('\r\033[J[{}] {}/{}, rate {:.3g}s, time {}, {} left, end {}'.format(
            bar, self.done, self.total, rate, hms(elapsed), hms(eta), end_time))
        if self.done >= self.total:
            sys.stdout.write('\n')
        sys.stdout.flush()
