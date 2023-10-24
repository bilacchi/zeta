def ensure_fit(func):
    def new_func(self, *args, **kwargs):
        assert self.isfit, f'{self.name}: Make sure to fit first'
        return func(self, *args, **kwargs)

    return new_func
