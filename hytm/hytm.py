
import math

import numpy as nmpy

from _strptime import TimeRE

from datetime import datetime, timedelta, timezone

import pytz
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as dtim_pars

from typing import Optional, ClassVar, Callable, Literal, Self
from dataclasses import dataclass, field

class HyTmly:

    defa_frmt: ClassVar[str] = "%Y%m%d%H%M%S"
    defa_regx: ClassVar[TimeRE] = TimeRE().compile(format=defa_frmt)

    veue_frmt: ClassVar[str] = "%Y-%m-%d %H:%M"

    freq_doma: ClassVar[dict] = {freq: dict(parm=parm, fkey=fkey) for freq, parm, fkey in zip(
        ["year", "month", "week", "day", "hour", "minute", "second"],
        ["years", "months", "weeks", "days", "hours", "minutes", "seconds"],
        ["yr", "mo", "wk", "d", "h", "m", "s"],
    )}

    (year,month,week,day,hour,minute,second) = list(freq_doma.keys())

    dmin: ClassVar[datetime] = datetime(year=1970, month=1,  day=1,  hour=0, minute=0, second=0, tzinfo=timezone.utc)
    dmax: ClassVar[datetime] = datetime(year=9999, month=12, day=31, hour=0, minute=0, second=0, tzinfo=timezone.utc)

    isdt: ClassVar[Callable[[str],bool]] = lambda dstr: bool(HyTmly.dstr_regx.match(dstr))

    dnow: ClassVar[Callable[[], datetime]] = lambda: datetime.now(tz=timezone.utc)

    dstr: ClassVar[Callable[[datetime], str]] = lambda dtim: dtim and dtim.strftime(HyTmly.defa_frmt) or str()
    sfnd: ClassVar[Callable] = lambda strn, subs: strn if 0 >= strn.find(subs) else strn[:strn.find(subs)]
    sdat: ClassVar[Callable[[datetime], datetime]] = lambda date: next((datetime.fromisoformat(f'{HyTmly.sfnd(isod, "+")}+00:00') for isod in [date.isoformat()]))

    pars: ClassVar[Callable[[str], datetime]] = lambda dstr: HyTmly.sdat(dtim_pars(dstr, fuzzy=True))

    tmst: ClassVar[Callable[[datetime], int]] = lambda dtim: int(f'{int(dtim.timestamp())}000')
    ftms: ClassVar[Callable[[int], datetime]] = lambda tmst: datetime.fromtimestamp(int(str(tmst)[:10]), tz=timezone.utc)

    dlts: ClassVar[Callable[[timedelta], int]] = lambda dlta: (dlta.days * 24 * 60 * 60) + dlta.seconds

    def __iter__(self):
        for freq in list(HyTmly.freq_doma):
            yield freq


@dataclass
class HyDelta:
    intv: int = 1
    freq: str = HyTmly.minute

    dlta: relativedelta = field(default=None, init=False)

    def __post_init__(self):
        #self.dlta = self.dlta or HyTmly.dnow() + relativedelta(**{HyTmly.freq_doma[self.freq]["parm"]: self.intv}) - HyTmly.dnow()
        self.dlta = relativedelta(**{HyTmly.freq_doma[self.freq]["parm"]: self.intv})

    def make_dict(self) -> dict:
        return dict(intv=self.intv, freq=self.freq)

    def __len__(self):
        return HyTmly.dlts((HyTmly.dmin + self.dlta) - HyTmly.dmin)

    def __str__(self):
        return f'{self.intv}{HyTmly.freq_doma[self.freq]["fkey"]}'

    def __call__(self, tcks: int = 1) -> timedelta:
        return self.dlta * tcks

    def pars_intv(self):
        return f'{self.intv}{HyTmly.freq_doma[self.freq]["fkey"]}'

    def cont_tcks(self, dlta = timedelta):
        return int(math.floor(HyTmly.dlts(dlta) / len(self)))


@dataclass
class HySpan:
    bdat: str | datetime = HyTmly.dmin
    edat: str | datetime = HyTmly.dnow()
    ctof: Optional[datetime] = field(default=None)

    dlta: HyDelta = field(default_factory=HyDelta)
    snip: bool = field(default=False)

    cdlt: HyDelta = field(default=None, init=False)
    sdlt: HyDelta = field(default=None, init=False)

    ssep: ClassVar[str] = field(default=" | ", init=False)

    def __post_init__(self):
        (self.bdat, self.edat, self.ctof) = [date if not date else HyTmly.pars(date) if isinstance(date, str) else HyTmly.sdat(date) for date in [self.bdat, self.edat, self.ctof]]
        self.ctof = self.edat if not self.ctof or ( self.edat <= self.ctof or self.bdat >= self.ctof) else self.ctof

    def __snap(self, bdat: str | datetime = None, edat: str | datetime = None, ctof: str | datetime = None ):
        self.bdat = HyTmly.pars(bdat) if isinstance(bdat, str) else HyTmly.sdat(bdat) if bdat else self.bdat
        self.ctof = HyTmly.pars(ctof) if isinstance(ctof, str) else HyTmly.sdat(ctof) if bdat else self.ctof
        self.edat = HyTmly.pars(edat) if isinstance(edat, str) else HyTmly.sdat(edat) if edat else self.edat
        bdat, ctof, edat = self.bdat.replace(second=0, microsecond=0), self.ctof.replace(second=0, microsecond=0), self.edat.replace(second=0, microsecond=0)
        (self.bdat, self.ctof, self.edat) = [HyTmly.sdat(datetime(year=bdat.year, month=1, day=1)), HyTmly.sdat(datetime(year=ctof.year, month=1, day=1)), HyTmly.sdat(datetime(year=edat.year + 1, month=1, day=1))]
        self.bdat -= (self.dlta() * int(HyTmly.dlts(self.bdat - bdat) // len(self.dlta)))
        self.ctof -= (self.dlta() * int(HyTmly.dlts(self.ctof - ctof) // len(self.dlta)))
        self.edat -= (self.dlta() * int(HyTmly.dlts(self.edat - edat) // len(self.dlta)))
        if self.edat > HyTmly.dnow():
            self.edat -= self.dlta()
            self.bdat -= self.dlta()
        return self

    def snap(self, bdat: str | datetime = None, edat: str | datetime = None, ctof: str | datetime = None ):
        self.bdat = HyTmly.pars(bdat) if isinstance(bdat, str) else HyTmly.sdat(bdat) if bdat else self.bdat
        self.ctof = HyTmly.pars(ctof) if isinstance(ctof, str) else HyTmly.sdat(ctof) if bdat else self.ctof
        self.edat = HyTmly.pars(edat) if isinstance(edat, str) else HyTmly.sdat(edat) if edat else self.edat
        bdat, ctof, edat = self.bdat.replace(second=0, microsecond=0), self.ctof.replace(second=0, microsecond=0), self.edat.replace(second=0, microsecond=0)
        (self.bdat, self.ctof, self.edat) = [HyTmly.sdat(datetime(year=bdat.year, month=1, day=1)), HyTmly.sdat(datetime(year=ctof.year, month=1, day=1)), HyTmly.sdat(datetime(year=edat.year + 1, month=1, day=1))]
        self.bdat -= (self.dlta() * int(HyTmly.dlts(self.bdat - bdat) // len(self.dlta)))
        self.ctof -= (self.dlta() * int(HyTmly.dlts(self.ctof - ctof) // len(self.dlta)))
        self.edat -= (self.dlta() * int(HyTmly.dlts(self.edat - edat) // len(self.dlta)))
        if self.edat > HyTmly.dnow():
            self.edat -= self.dlta()
            self.bdat -= self.dlta()
        return self

    def slide(self, stride: int | datetime = None):
        if self.snip and stride and isinstance(stride, int|datetime):
            dlta = self.dlta(stride) if isinstance(stride, int) else (stride.replace(second=0, microsecond=0) - self.edat) + self.dlta()
            self.bdat, self.ctof, self.edat = [dtim + dlta for dtim in (self.bdat, self.ctof, self.edat)]
        return self

    def split(self, ctof: datetime = None):
        ctof = ctof or self.ctof
        return (self, None) if not ctof or ctof == self.edat else (HySpan(bdat=self.bdat, edat=ctof, dlta=self.dlta), HySpan(bdat=ctof + self.dlta(), edat=self.edat, dlta=self.dlta))

    def __call__(self, dtim: datetime):
        return self.bdat <= dtim <= self.edat

    def __len__(self):
        return int(math.floor(HyTmly.dlts(self.edat - self.bdat) // len(self.dlta)) + 1)

    def __iter__(self):
        bdat = self.bdat
        while bdat <= self.edat:
            yield bdat
            bdat += self.dlta()
        """
        for idex in range(len(self)):
            yield min(self.bdat + (self.dlta() * idex), self.edat)
        """

    def __str__(self):
        return HySpan.ssep.join([date.strftime("%Y-%m-%d %H:%M") for date in [self.bdat, self.edat]])

    def __gt__(self, other: Self)  -> bool:
        return self.bdat > other.bdat

    def __ge__(self, other: Self)  -> bool:
        return self.bdat >= other.bdat

    def __lt__(self, other: Self)  -> bool:
        return self.bdat < other.bdat

    def __le__(self, other: Self)  -> bool:
        return self.bdat <= other.bdat

    def __sub__(self, tcks: int):
        if tcks:
            self.bdat = (self.edat - (self.dlta(tcks)))
        return self

    def __add__(self, tcks: int):
        if tcks:
            self.edat = (self.edat + (self.dlta(tcks)))
        return self

    def cycle(self, cycl: HyDelta = HyDelta(intv=1, freq=HyTmly.day), step: HyDelta = HyDelta(intv=1, freq=HyTmly.hour), sin_cos: bool = False):
        assert "week" not in (cycl.freq, step.freq)
        init = HySpan(bdat=HyTmly.pars(f"{self.bdat.year}-01-01 00:00:00"), edat=self.edat, dlta=self.dlta)
        bdat = init.bdat
        outs = nmpy.ndarray((0,2)) if sin_cos else nmpy.ndarray((0,1))
        while bdat <= self.edat:
            arng = []
            cend = bdat + cycl()
            while bdat < cend:
                send = bdat + step()
                arng.append(nmpy.repeat([len(arng)], HyTmly.dlts(send - bdat) // len(self.dlta)))
                bdat = send
            alen = len(arng)
            arng = [arct.reshape(arct.shape[0], 1) for arct in [nmpy.concatenate(arng)]].pop()
            outs = nmpy.concatenate([outs, arng] if not sin_cos else [outs, nmpy.concatenate(list(nmpy.around(fn(2 * nmpy.pi * arng / alen), 8) for fn in [nmpy.sin, nmpy.cos]), axis=-1)])
        return outs[HyTmly.dlts(self.bdat - init.bdat) // len(self.dlta):(-(HyTmly.dlts(bdat - self.edat) // len(self.dlta)) + 1) or None]

    def make_dict(self, frmt=HyTmly.defa_frmt) -> dict:
        return dict(bdat=self.bdat.strftime(frmt), edat=self.edat.strftime(frmt), grnl=self.dlta.make_dict())

if __name__ == "__main__":
    import random
    breakpoint()

    veri = [(HySpan()-random.randint(1000,2000)).snap() for _ in range(1024)]

    breakpoint()
    span = (HySpan().snap() - 10080).snap()
    span.slide(stride=-1274)
    span.slide(stride=HyTmly.dnow() + span.dlta(12960))
    breakpoint()

    def test():
        import random
        import matplotlib.pyplot as plot
        import matplotlib.colors as pcol

        def fast_plot(*args):
            plot.close()
            plot.figure(figsize=(24, 8), clear=True)
            colr = list(pcol.TABLEAU_COLORS)
            [plot.plot(data, color=colr.pop(random.randint(0, len(colr) - 1)), label="Series %s" % idex, linestyle=["solid", "dashed"][idex % 2]) for idex, data in enumerate(args)]
            plot.legend()
            plot.grid()
            plot.show()

        span = HySpan(bdat=HyTmly.pars("2017-02-14"), edat=HyTmly.pars("2024-01-16"), dlta=HyDelta(intv=1, freq=HyTmly.day))
        cycl = span.cycle(cycl=HyDelta(intv=1, freq=HyTmly.month), step=HyDelta(intv=1, freq=HyTmly.day), sin_cos=True)
        fast_plot(cycl)
        cycl = span.cycle(cycl=HyDelta(intv=1, freq=HyTmly.year), step=HyDelta(intv=1, freq=HyTmly.day))
        fast_plot(cycl)
    test()
    pass