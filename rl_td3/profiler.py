import cProfile, pstats, io


class Profiler(object):
    def __init__(self):
        self.pr = cProfile.Profile()

    def start_profile(self):
        self.pr.enable()

    def end_profile(self):
        self.pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s).sort_stats('cumtime')
        ps.print_stats(20)
        print(s.getvalue())
        ps = pstats.Stats(self.pr, stream=s).sort_stats('time')
        ps.print_stats(20)
        print(s.getvalue())
